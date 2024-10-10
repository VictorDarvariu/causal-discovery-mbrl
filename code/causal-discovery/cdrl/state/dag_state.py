import warnings
from copy import deepcopy
from itertools import product

import networkx as nx
import numpy as np
import xxhash

from cdrl.environment.graph_edge_env import EnvPhase
from cdrl.utils.graph_utils import nx_graph_to_adj_matrix, contains_cycles_exact

budget_eps = 1e-5


class DAGState(object):
    """
    This class implements the state for the decision-making process defined over DAGs. Actions decided by the agents modify this state.
    The state contains information about the graph topology (e.g., the intermediate graph structure as decided so far)
    and complex bookkeeping information that is required to determine which next actions are valid choices.

    It also implements the incremental algorithm for determining cycle-inducing candidate edges.
    """
    def __init__(self, g, init_tracking=True):
        self.num_nodes = g.number_of_nodes()
        self.node_labels = np.arange(self.num_nodes)
        self.all_nodes_set = set(self.node_labels)

        self.edge_list = sorted(g.edges())
        self.edge_set = set(self.edge_list)

        self.num_edges = len(self.edge_list)

        self.node_in_degrees = np.array([deg for (node, deg) in sorted(g.in_degree(), key=lambda deg_pair: deg_pair[0])])
        self.node_out_degrees = np.array([deg for (node, deg) in sorted(g.out_degree(), key=lambda deg_pair: deg_pair[0])])
        self.first_node = None

        # "dynamic edges" are not added to the topology straight away but stored; used extensively in the MCTS simulation policy.
        self.dynamic_edges = None

        self.last_added_edge = None

        if init_tracking:
            self.initialize_cycle_tracking()


    def initialize_cycle_tracking(self):
        """
        Starting condition for the incremental algorithm: initialize the tracking of cycles based on the starting graph topology by performing traversals.
        """
        nx_graph = self.to_networkx()

        self.cycle_ind_edges = {}
        self.node_ancestors = {n: set() for n in self.node_labels}
        self.node_descendants = {n: set() for n in self.node_labels}
        for n in self.node_labels:
            self.node_ancestors[n] = set(nx.ancestors(nx_graph, n))
            self.node_descendants[n] = set(nx.descendants(nx_graph, n))
        non_edges = list(nx.non_edges(nx_graph))
        for ne in non_edges:
            edge_from, edge_to = ne

            if not nx_graph.has_edge(edge_to, edge_from):
                graph_cp = deepcopy(nx_graph)
                graph_cp.add_edge(edge_from, edge_to)

                if contains_cycles_exact(graph_cp):
                    if edge_from not in self.cycle_ind_edges:
                        self.cycle_ind_edges[edge_from] = set()

                    self.cycle_ind_edges[edge_from].add(edge_to)


    def update_cycle_inducing_edges(self, last_added_edge):
        """
        Incremental algorithm that updates the cycle-inducing candidate edges based on the previous known cycle-inducing edges and the last edge that was added.
        """
        if self.cycle_ind_edges is None:
            self.initialize_cycle_tracking()
        else:
            last_from, last_to = last_added_edge

            # the reverse of an edge that produces a cycle may have been added in the previous step.
            # as such, the edge becomes invalid via the directional connectivity constraint rather than acyclicity.
            if last_to in self.cycle_ind_edges and last_from in self.cycle_ind_edges[last_to]:
                self.cycle_ind_edges[last_to].remove(last_from)

            A_and_ancestors = self.node_ancestors[last_from].union({last_from})
            B_and_descendants = self.node_descendants[last_to].union({last_to})

            newly_cycle_inducing = set(product(B_and_descendants, A_and_ancestors))
            newly_cycle_inducing.remove((last_to, last_from))  # trivially true.

            for ne in newly_cycle_inducing:
                edge_from, edge_to = ne

                if edge_from not in self.cycle_ind_edges:
                    self.cycle_ind_edges[edge_from] = set()

                edge_present_dynamic = (self.dynamic_edges is not None and (edge_to, edge_from) in self.dynamic_edges)
                if not (self.has_edge(edge_to, edge_from) or edge_present_dynamic):
                    if edge_to not in self.cycle_ind_edges[edge_from]:
                        self.cycle_ind_edges[edge_from].add(edge_to)
                        if hasattr(self, "latest_disallowed_edges"):
                            self.latest_disallowed_edges.add((edge_from, edge_to))

            # update ancestors and descendants now.
            for n in A_and_ancestors:
                self.node_descendants[n].update(B_and_descendants)

            for n in B_and_descendants:
                self.node_ancestors[n].update(A_and_ancestors)


    def has_edge(self, first_node, second_node):
        """Checks whether the edge already exists."""
        return (first_node, second_node) in self.edge_set

    def add_edge(self, first_node, second_node):
        """Adds a new edge to the graph and updates bookkeeping information. Returns a new state."""
        new_g = self.copy()

        new_g.edge_list.append((first_node, second_node))
        new_g.edge_set.add((first_node, second_node))

        new_g.node_out_degrees[first_node] += 1
        new_g.node_in_degrees[second_node] += 1
        new_g.last_added_edge = (first_node, second_node)

        return new_g, 1

    def remove_edge(self, first_node, second_node):
        """Removes an edge from the graph and updates bookkeeping information. Returns a new state."""
        new_g = self.copy()

        new_g.edge_list.remove((first_node, second_node))
        new_g.edge_set.remove((first_node, second_node))

        new_g.node_out_degrees[first_node] -= 1
        new_g.node_in_degrees[second_node] -= 1
        new_g.last_added_edge = None

        return new_g, 1

    def add_edge_dynamically(self, first_node, second_node):
        """Adds edge dynamically, i.e., keeps track of this edge as to-be-added but does not modify the edge set itself."""
        self.latest_disallowed_edges = set()

        self.dynamic_edges.append((first_node, second_node))
        self.node_out_degrees[first_node] += 1
        self.node_in_degrees[second_node] += 1
        self.update_cycle_inducing_edges((first_node, second_node))

        self.latest_disallowed_edges.add((second_node, first_node))
        self.latest_disallowed_edges.add((first_node, second_node))
        return 1

    def remove_edge_dynamically(self, first_node, second_node):
        """Removes edge dynamically."""
        self.dynamic_edges.append((first_node, second_node))
        self.node_out_degrees[first_node] -= 1
        self.node_in_degrees[second_node] -= 1
        return 1

    def populate_banned_actions(self, phase, budget=None, enforce_acyclic=True):
        """Populates actions that are forbidden based on the MDP definition."""
        if budget is not None:
            if budget < budget_eps:
                self.banned_actions = self.all_nodes_set
                return

        if enforce_acyclic and self.first_node is None:
            if self.dynamic_edges is not None and len(self.dynamic_edges) > 0:
                last_edge = self.dynamic_edges[-1]
            else:
                last_edge = self.last_added_edge

            if last_edge is not None:
                self.update_cycle_inducing_edges(last_edge)

        if self.first_node is None:
            self.banned_actions = self.get_invalid_first_nodes(phase, enforce_acyclic=enforce_acyclic)
        else:
            self.banned_actions = self.get_invalid_edge_ends(phase, self.first_node, enforce_acyclic=enforce_acyclic)

    def get_invalid_first_nodes(self, phase, enforce_acyclic=True):
        """Determines the nodes that are not valid edge stubs based on the MDP definition."""
        if phase == EnvPhase.CONSTRUCT:
            max_connected_nodes = set([node_id for node_id in self.node_labels if self.node_out_degrees[node_id] + self.node_in_degrees[node_id] == (self.num_nodes - 1)])
            if not enforce_acyclic:
                return max_connected_nodes

            # find the nodes whose current out_degree plus in_degree plus number of cycle-inducing edges is equal to the number of nodes - 1.
            # these should additionally be banned as you cannot build an edge starting from them.
            # in_degree is needed to account for incoming edges, since we cannot add the reverse ones.

            if self.cycle_ind_edges is not None:
                for node in self.cycle_ind_edges.keys():
                    n_cyc_inducing_edges = len(self.cycle_ind_edges[node])
                    out_degree = self.node_out_degrees[node]
                    in_degree = self.node_in_degrees[node]

                    n_disallowed_connections = n_cyc_inducing_edges + out_degree + in_degree

                    if n_disallowed_connections == self.num_nodes - 1:
                        max_connected_nodes.add(node)

            return max_connected_nodes
        else:
            no_outgoing_edges = set([node_id for node_id in self.node_labels if self.node_out_degrees[node_id] == 0])
            return no_outgoing_edges


    def get_invalid_edge_ends(self, phase, query_node, enforce_acyclic=True):
        """Determines the nodes that cannot be edge endpoints for an edge starting at the query_node stub. This is based on the MDP definition."""
        results = set()
        results.add(query_node)

        if phase == EnvPhase.CONSTRUCT:
            results.update({edge_from for (edge_from, edge_to) in self.edge_set if edge_to == query_node})
            results.update({edge_to for (edge_from, edge_to) in self.edge_set if edge_from == query_node})

            if self.dynamic_edges is not None:
                dynamic_left = [entry[0] for entry in self.dynamic_edges if entry[1] == query_node]
                results.update(dynamic_left)
                dynamic_right = [entry[1] for entry in self.dynamic_edges if entry[0] == query_node]
                results.update(dynamic_right)

            if enforce_acyclic and self.cycle_ind_edges is not None and query_node in self.cycle_ind_edges:
                results.update(self.cycle_ind_edges[query_node])

        else:
            nodes_existing = {edge_to for (edge_from, edge_to) in self.edge_set if edge_from == query_node}
            invalid_ends = self.all_nodes_set - nodes_existing
            results.update(invalid_ends)

            if self.dynamic_edges is not None:
                already_removed = set([entry[1] for entry in self.dynamic_edges if entry[0] == query_node])
                results.update(already_removed)


        return results

    def init_dynamic_edges(self):
        """Initialize the dynamic edges."""
        self.dynamic_edges = []

    def apply_dynamic_edges(self, phase):
        """Apply the dynamic edges: create a new state with the dynamic edges actually added to the graph topology."""
        nx_graph = self.to_networkx()
        for edge in self.dynamic_edges:
            if phase == EnvPhase.CONSTRUCT:
                nx_graph.add_edge(edge[0], edge[1])
            else:
                nx_graph.remove_edge(edge[0], edge[1])

        final_graph = DAGState(nx_graph, init_tracking=False)
        return final_graph

    def to_networkx(self):
        """Convert to networkx DiGraph object."""
        edges = self.edge_list
        g = nx.DiGraph()
        g.add_nodes_from(self.node_labels)
        g.add_edges_from(edges)
        return g

    def get_edge_list(self):
        """Retrieve the edge list."""
        return self.edge_list

    def get_edges_with_aliases(self, node_aliases):
        """Retrieve the edge list, replacing node IDs with their aliases."""
        return [(node_aliases[e[0]], node_aliases[e[1]]) for e in self.edge_list]

    def display(self, ax=None):
        """Draw the DAG."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw_circular(nx_graph, with_labels=True, ax=ax)

    def display_with_positions(self, node_positions, ax=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw(nx_graph, pos=node_positions, with_labels=True, ax=ax)

    def draw_to_file(self, filename):
        """Draw the DAG to a file."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_size_length = self.num_nodes
        figsize = (fig_size_length, fig_size_length)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        self.display(ax=ax)
        fig.savefig(filename)
        plt.close()

    def get_adjacency_matrix(self):
        """Convert to numpy adjacency matrix."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            return nx_graph_to_adj_matrix(nx_graph)

    def copy(self):
        """Create a deepcopy."""
        return deepcopy(self)


def get_graph_hash(g, size=64, include_first=False):
    """Hash the graph based on the graph topology and edge stub (if it exists)."""
    if size == 32:
        hash_instance = xxhash.xxh32()
    elif size == 64:
        hash_instance = xxhash.xxh64()
    else:
        raise ValueError("only 32 or 64-bit hashes supported.")

    if include_first:
        if g.first_node is not None:
            hash_instance.update(np.array([g.first_node]))
        else:
            hash_instance.update(np.zeros(g.num_nodes))

    sorted_edges = np.array(sorted(g.edge_list))
    hash_instance.update(sorted_edges)
    graph_hash = hash_instance.intdigest()
    return graph_hash

