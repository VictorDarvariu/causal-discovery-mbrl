import networkx as nx

from cdrl.environment.graph_edge_env import EnvPhase
from cdrl.state.dag_state import get_graph_hash, DAGState
from cdrl.state.instance_generators import BNLearnInstanceGenerator


def test_discrete_instance_generation():
    inst_name = "asia"
    n = 1000
    disc_inst = BNLearnInstanceGenerator.get_instance(instance_name=inst_name, n=n)

    dag_state = disc_inst.start_state

    aliases = disc_inst.node_aliases
    adj_matrix = disc_inst.true_adj_matrix

    assert disc_inst.true_num_edges == 8
    assert disc_inst.d == 8

    true_edges = [("asia", "tub"), ("tub", "either"), ("either", "xray"), ("either", "dysp"),
                  ("bronc", "dysp"), ("lung", "either"), ("smoke", "lung"), ("smoke", "bronc")]

    assert disc_inst.datasize == n

    for (start, end) in true_edges:
        start_idx = aliases.index(start)
        end_idx = aliases.index(end)

        assert adj_matrix[start_idx, end_idx] == 1

