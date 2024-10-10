import numpy as np
from scipy.linalg import expm as matrix_exponential
import networkx as nx

CYCNESS_THRESHOLD = 1e-5

def compute_cycness(adj_matrix):
    """NOTEARS acyclicity characterisation."""
    maxlen = adj_matrix.shape[0]
    cycness = np.trace(matrix_exponential(np.array(adj_matrix))) - maxlen
    return cycness


def contains_cycles(adj_matrix):
    """Determines cycle existence based on NOTEARS acyclicity characterisation."""
    cycness = compute_cycness(adj_matrix)
    return cycness > CYCNESS_THRESHOLD


def contains_cycles_exact(nx_graph):
    """Determines cycle existence based on traversals."""
    return (not nx.algorithms.dag.is_directed_acyclic_graph(nx_graph))


def nx_graph_from_adj_matrix(adj_matrix):
    """Converts adjacency matrix to networkx graph."""
    return nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)


def nx_graph_to_adj_matrix(nx_graph, nodelist=None):
    """Converts networkx graph to adjacency matrix."""
    if nodelist is None:
        nodelist = np.arange(nx_graph.number_of_nodes())
    return np.asarray(nx.convert_matrix.to_numpy_array(nx_graph, nodelist=nodelist, dtype=np.int32))


def edge_list_from_adj_matrix(adj_matrix):
    """Converts adjacency matrix to edge list."""
    nonzero_idx = np.transpose(np.nonzero(adj_matrix))
    edge_list = [tuple(row) for row in nonzero_idx]
    return edge_list

def edge_list_to_nx_graph(edge_list, num_nodes):
    """Converts an edge list to a networkx graph."""
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(edge_list)
    return nx_graph