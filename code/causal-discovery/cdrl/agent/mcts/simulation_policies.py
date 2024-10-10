from copy import copy, deepcopy
from random import Random

import networkx as nx

from cdrl.state.dag_state import get_graph_hash
from cdrl.utils.graph_utils import contains_cycles_exact


class SimulationPolicy(object):
    """
    Base class for simulation policies.
    """
    def __init__(self, local_random, **kwargs):
        self.local_random = local_random

    def get_random_state(self):
        return self.local_random.getstate()


class RandomSimulationPolicy(SimulationPolicy):
    def __init__(self,  local_random, **kwargs):
        super().__init__(local_random)

    def choose_action(self, state, possible_actions, current_depth):
        available_acts = tuple(possible_actions)
        chosen_action = self.local_random.choice(available_acts)
        return chosen_action

    def reset_for_new_simulation(self, start_state):
        pass


class NaiveSimulationPolicy(SimulationPolicy):
    """
    Also implements a uniform random simulation policy,
    but uses a naive search to determine the valid next actions instead of the incremental algorithm.
    """
    def __init__(self,  local_random, **kwargs):
        super().__init__(local_random, **kwargs)
        self.next_action = None

    def choose_action(self, state, possible_actions, current_depth):
        """As with some of the BaselineAgent subclasses, need to reconcile choosing an edge (two nodes) with the MDP definition
        implemented by the environment allows choosing a single node per timestep.
        """
        if current_depth % 2 == 0:
            chosen_edge = self.sample_valid_edge(state)
            first_node = chosen_edge[0]
            second_node = chosen_edge[1]
            self.next_action = second_node
            return first_node
        else:
            if self.next_action is None:
                first_node = state.first_node

                if first_node is None:
                    raise ValueError(f"first_node shouldn't be None!")

                chosen_edge = self.sample_valid_edge(state, first_node=first_node)
                return chosen_edge[1]
            else:
                next_act = copy(self.next_action)
                self.next_action = None
                return next_act

    def sample_valid_edge(self, state, first_node=None):
        """Samples an edge out of the valid candidates, performing explicit traversals to determine if the addition of an edge would cause a cycle."""
        chosen_edge = None

        non_edges = deepcopy(self.start_non_edges)
        for dyn_edge in state.dynamic_edges:
            non_edges.remove((dyn_edge[0], dyn_edge[1]))
            non_edges.remove((dyn_edge[1], dyn_edge[0]))

        ne_choices = list(non_edges)
        if first_node is not None:
            ne_choices = [ne for ne in ne_choices if ne[0] == first_node]

        self.local_random.shuffle(ne_choices)
        ne_idx = 0

        while True:
            possible_ne = ne_choices[ne_idx]
            g_cp = deepcopy(self.start_nx_graph)
            g_cp.add_edges_from(state.dynamic_edges)
            g_cp.add_edge(possible_ne[0], possible_ne[1])
            # carry out traversals to determine cycle existence
            if not contains_cycles_exact(g_cp):
                chosen_edge = possible_ne
                break
            else:
                ne_idx += 1


        return chosen_edge



    def reset_for_new_simulation(self, start_state):
        """Prepares the simulation policy for executing a new simulation."""
        self.local_random = Random()
        self.local_random.seed(get_graph_hash(start_state))

        self.start_nx_graph = start_state.to_networkx()

        non_edges = set(nx.non_edges(self.start_nx_graph))
        self.start_non_edges = {ne for ne in non_edges if not self.start_nx_graph.has_edge(ne[1], ne[0])}

