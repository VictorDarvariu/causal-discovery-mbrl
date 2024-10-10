from copy import deepcopy
from itertools import product

import numpy as np

from cdrl.state.dag_state import DAGState
from cdrl.utils.config_utils import local_seed
from cdrl.utils.graph_utils import edge_list_from_adj_matrix, check_contains_undirected, split_directed_undirected, nx_graph_from_adj_matrix


def extract_validation_perf_from_metrics_dict(metrics_dict):
    """Specifies the metric to use for validation -- only the score function is
    valid, since choosing another metric relies on knowledge of the ground truth."""
    if "prune_mdp" in metrics_dict["results"]:
        return -metrics_dict["results"]["prune_mdp"]["reward"]
    else:
        return -metrics_dict["results"]["construct"]["reward"]


def get_metrics_dict(construct_output, prune_output, disc_inst, reward_function, include_cam_pruning=False):
    """Evalutes the metrics for an environment run that may include both construction and pruning."""
    eval_dict = {}
    eval_dict["instance_metadata"] = dict(disc_inst.instance_metadata._asdict())

    results_dict = {}

    if construct_output is not None:
        results_dict["construct"] = get_metrics_for_graph(disc_inst, construct_output[0][0], reward_function)

    if prune_output is not None:
        results_dict["prune_mdp"] = get_metrics_for_graph(disc_inst, prune_output[0][0], reward_function)

    if include_cam_pruning:
        if construct_output is not None:
            results_dict["prune_cam"] = get_metrics_for_graph(disc_inst, construct_output[0][0], reward_function, prune_with_cam=True)
        else:
            results_dict["prune_cam"] = get_metrics_for_graph(disc_inst, disc_inst.start_state, reward_function, prune_with_cam=True)

    gt_data = count_accuracy(disc_inst.true_adj_matrix, disc_inst.true_adj_matrix)
    true_edges = edge_list_from_adj_matrix(disc_inst.true_adj_matrix)
    gt_data["edges"] = true_edges

    if disc_inst.true_graph is not None:
        gt_reward = reward_function.calculate_reward_single_graph(disc_inst.true_graph)
        gt_data["reward"] = gt_reward
        results_dict["ground_truth"] = gt_data

    eval_dict["results"] = results_dict

    return eval_dict


def get_metrics_for_graph(disc_inst, graph, reward_function, prune_with_cam=False):
    """Computes the metrics for a given problem instance and graph found by a causal discovery procedure."""
    discovered_graph = graph
    discovered_adj = discovered_graph.get_adjacency_matrix()
    discovered_adj_ji = discovered_adj.T

    if prune_with_cam:
        from cdrl.utils.pruning import graph_prunned_by_coef, graph_prunned_by_coef_2nd
        from cdrl.utils.cam_with_pruning_cam import pruning_cam

        reg_type = disc_inst.instance_metadata.reg_type
        if reg_type == 'LR':
            # RL-BIC pruning code expects j->i adjacency matrix, and returns j->i adjacency matrix. Hence, pass j->i and transpose the result.
            discovered_adj = np.array(graph_prunned_by_coef(discovered_adj_ji, disc_inst.inputdata), dtype=np.int32).T
        elif reg_type == 'QR':
            # RL-BIC pruning code expects j->i adjacency matrix, and returns j->i adjacency matrix. Hence, pass j->i and transpose the result.
            discovered_adj = np.array(graph_prunned_by_coef_2nd(discovered_adj_ji, disc_inst.inputdata), dtype=np.int32).T
        elif reg_type == 'GPR':
            # CAM pruning code expects i->j and returns i->j, so no need to transpose.
            discovered_adj = pruning_cam(disc_inst.inputdata, discovered_adj)


    has_undirected_edges = check_contains_undirected(discovered_adj)
    if has_undirected_edges:
        # if there are undirected edges, need to split the
        # adjacency matrix into "directed" and "undirected" components
        # that are then passed to the metrics computation function.
        adj_directed, adj_undirected = split_directed_undirected(discovered_adj)
        metrics_out = count_accuracy(disc_inst.true_adj_matrix, adj_directed, B_und=adj_undirected)

        num_directed = np.count_nonzero(adj_directed)
        num_undirected = np.count_nonzero(adj_undirected)

        frac_undirected = num_undirected / (num_undirected + num_directed)
        # additionally record fraction of undirected edges.
        metrics_out["fue"] = frac_undirected


    else:
        metrics_out = count_accuracy(disc_inst.true_adj_matrix, discovered_adj)

    graph_for_computing_reward = discovered_graph if not prune_with_cam else (DAGState(nx_graph_from_adj_matrix(discovered_adj)))
    metrics_out["reward"] = reward_function.calculate_reward_single_graph(graph_for_computing_reward)

    predicted_edges = edge_list_from_adj_matrix(discovered_adj)
    metrics_out["edges"] = predicted_edges

    return metrics_out


def count_accuracy(B_true, B, B_und=None) -> dict:
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.

    Args:
        B_true: ground truth graph
        B: predicted graph
        B_und: predicted undirected edges in CPDAG, asymmetric

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive

    This code was adapted unchanged from the RLBIC repository, whose authors themselves adapted it from the authors of NOTEARS.
    """

    if B_true is None:
        # true adjacency matrix not known; report only number of output edges.
        pred = np.flatnonzero(B)
        pred_size = len(pred)
        return {'pred_size': pred_size}

    d = B.shape[0]
    # linear index of nonzeros
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)

    if B_und is not None:
        B_lower = np.add(B_lower, np.tril(B_und + B_und.T), out=B_lower, casting="unsafe")
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    acc_res = {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'pred_size': pred_size}
    return acc_res

def generate_search_space(parameter_grid,
                          random_search=False,
                          random_search_num_options=20,
                          random_search_seed=42):
    """Generates combinations of hyperparameters from their possible values, optionally subsetting them randomly."""
    combinations = list(product(*parameter_grid.values()))
    search_space = {i: combinations[i] for i in range(len(combinations))}

    if random_search:
        if not random_search_num_options > len(search_space):
            reduced_space = {}
            with local_seed(random_search_seed):
                random_indices = np.random.choice(len(search_space), random_search_num_options, replace=False)
                for random_index in random_indices:
                    reduced_space[random_index] = search_space[random_index]
            search_space = reduced_space
    return search_space


def construct_search_spaces(experiment_conditions):
    """Generates combinations of parameters for all objective functions and agents that form an experiment."""
    parameter_search_spaces = {}
    objective_functions = experiment_conditions.objective_functions

    for obj_fun in objective_functions:
        parameter_search_spaces[obj_fun.name] = {}
        for agent in experiment_conditions.agents:
            if agent.algorithm_name in experiment_conditions.hyperparam_grids[obj_fun.name]:
                agent_grid = experiment_conditions.hyperparam_grids[obj_fun.name][agent.algorithm_name]
                combinations = list(product(*agent_grid.values()))
                search_space = {}
                for i in range(len(combinations)):
                    k = str(i)
                    v = dict(zip(list(agent_grid.keys()), combinations[i]))
                    search_space[k] = v
                parameter_search_spaces[obj_fun.name][agent.algorithm_name] = search_space

    return parameter_search_spaces

