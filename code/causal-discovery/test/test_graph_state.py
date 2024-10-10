import networkx as nx

from cdrl.environment.graph_edge_env import EnvPhase
from cdrl.state.dag_state import get_graph_hash, DAGState
from cdrl.state.instance_generators import HardcodedInstanceGenerator


def test_dynamic_edges_cycle():
# def main():
    N = 6
    starting_dag = nx.DiGraph()
    starting_dag.add_nodes_from(list(range(N)))
    starting_dag.add_edges_from([(0, 1), (1, 2), (4, 5), (5, 0)])
    state = DAGState(starting_dag)
    state.populate_banned_actions(EnvPhase.CONSTRUCT, enforce_acyclic=True)
    assert len(state.banned_actions) == 0

    state.init_dynamic_edges()
    state.add_edge_dynamically(2, 3)
    state.populate_banned_actions(EnvPhase.CONSTRUCT, enforce_acyclic=True)
    assert (2 in state.banned_actions and 3 in state.banned_actions)

    state.add_edge_dynamically(0, 2)
    state.add_edge_dynamically(0, 3)
    state.populate_banned_actions(EnvPhase.CONSTRUCT, enforce_acyclic=True)
    assert (0 in state.banned_actions and 2 in state.banned_actions and 3 in state.banned_actions)

    end_state = state.apply_dynamic_edges(EnvPhase.CONSTRUCT)
    assert end_state.has_edge(2, 3)
    assert end_state.has_edge(0, 2)
    assert end_state.has_edge(0, 3)


def test_cycle_inducing_banned():
    N = 6
    starting_dag = nx.DiGraph()
    starting_dag.add_nodes_from(list(range(N)))
    starting_dag.add_edges_from([(0, 1), (1, 2), (4, 5), (5, 0), (2, 3)])
    state = DAGState(starting_dag)

    state.populate_banned_actions(EnvPhase.CONSTRUCT, enforce_acyclic=True)
    assert (2 in state.banned_actions and 3 in state.banned_actions)

    state.first_node = 0
    state.populate_banned_actions(EnvPhase.CONSTRUCT, enforce_acyclic=True)
    assert (4 in state.banned_actions)

def test_graph_hash():
    inst_name = "sachs"
    disc_inst = HardcodedInstanceGenerator.get_instance(instance_name=inst_name, normalize_data=True, transpose=False)
    empty_hash = get_graph_hash(disc_inst.start_state, size=64, include_first=True)

    other_state, _ = disc_inst.start_state.add_edge(0, 1)
    other_state, _ = other_state.add_edge(2, 1)
    other_state, _ = other_state.add_edge(5, 4)
    other_state, _ = other_state.add_edge(0, 5)
    nonempty_hash = get_graph_hash(other_state, size=64, include_first=True)
    assert empty_hash != nonempty_hash

    same_state_diff_order, _ = disc_inst.start_state.add_edge(0, 5)
    same_state_diff_order, _ = same_state_diff_order.add_edge(5, 4)
    same_state_diff_order, _ = same_state_diff_order.add_edge(2, 1)
    same_state_diff_order, _ = same_state_diff_order.add_edge(0, 1)

    ss_nonempty_hash = get_graph_hash(same_state_diff_order, size=64, include_first=True)
    assert nonempty_hash == ss_nonempty_hash

    other_state.first_node = 5
    fn_selected_hash = get_graph_hash(other_state, size=64, include_first=True)
    assert nonempty_hash != fn_selected_hash
