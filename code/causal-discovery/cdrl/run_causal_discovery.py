import argparse
import json
import pprint
import sys
from pathlib import Path

import networkx as nx


d = Path(__file__).resolve().parents[1]
sys.path.append(str(d.absolute()))
from pathlib import Path

from cdrl.utils.general_utils import NpEncoder
from cdrl.state.dag_state import DAGState
from cdrl.utils.graph_utils import edge_list_to_nx_graph
from cdrl.agent.mcts.mcts_agent import MonteCarloTreeSearchAgent
from cdrl.environment.graph_edge_env import DirectedGraphEdgeEnv, EnvPhase
from cdrl.evaluation.eval_utils import get_metrics_dict
from cdrl.reward_functions.reward_continuous_vars import ContinuousVarsBICRewardFunction
from cdrl.state.instance_generators import DiscoveryInstance, InstanceMetadata


def run_causal_discovery(args):
    dataset_path = Path("/experiment_data", args.dataset_file)

    gt_known = (args.gt_file is not None)

    gt_path = Path("/experiment_data", args.gt_file) if gt_known else None

    inst_name = dataset_path.stem

    instance_metadata = InstanceMetadata(name=inst_name, rvar_type="continuous", transpose=False, root_path=dataset_path.parent, reg_type=args.reg_type, rlbic_num_edges=-1)
    disc_inst = DiscoveryInstance(instance_metadata,
                                  data_path=str(dataset_path),
                                  dag_path=str(gt_path) if gt_known else None,
                                  normalize_data=args.normalize_data,
                                  starting_graph_generation="scratch",
                                  subsample_data=False
                                  )
    rfun = ContinuousVarsBICRewardFunction(disc_inst, penalise_cyclic=True, score_type=args.score_type, store_scores=True, reg_type=args.reg_type)

    initial_edge_budgets = {
        "construct": args.edge_budget,
        "prune": 0
    }

    env = DirectedGraphEdgeEnv(disc_inst, rfun, initial_edge_budgets=initial_edge_budgets, enforce_acyclic=True)

    agent = MonteCarloTreeSearchAgent(env)
    opts = {
            "random_seed": args.random_seed,
    }
    hyperparams = {
        'C_p': args.C_p,
        'adjust_C_p': args.adjust_C_p,
        'final_action_strategy': args.final_action_strategy,
        'expansion_budget_modifier': args.expansion_budget_modifier,
        'sim_policy': args.sim_policy,
        'rollout_depth': args.rollout_depth,
        'sims_per_expansion': args.sims_per_expansion,
        'transpositions_enabled': args.transpositions_enabled,
        'btm': args.btm
    }

    agent.setup(opts, hyperparams)
    construct_output = agent.eval([disc_inst.start_state], EnvPhase.CONSTRUCT)
    prune_output = None

    include_cam_pruning = args.include_cam_pruning
    eval_dict = get_metrics_dict(construct_output, prune_output, disc_inst, rfun, include_cam_pruning=include_cam_pruning)
    # pprint.pprint(eval_dict)


    output_dir = Path("/experiment_data", args.output_directory)
    output_dir.mkdir(exist_ok=True, parents=True)

    del eval_dict["instance_metadata"]
    vars_dict = vars(args)
    eval_dict["experiment_parameters"] = vars_dict

    results_file = output_dir / f"{inst_name}_results.json"

    with open(results_file, "w") as fh:
        json.dump(eval_dict, fh, indent=4, sort_keys=True, cls=NpEncoder)

    drawing_file = output_dir / f"{inst_name}_discovered_graph.pdf"
    final_state = construct_output[0][0]
    final_state.draw_to_file(drawing_file)

    if args.include_cam_pruning:
        drawing_file_cam = output_dir / f"{inst_name}_discovered_graph_pruned.pdf"
        post_pruning_edges = eval_dict['results']['prune_cam']['edges']
        post_pruning_state = DAGState(edge_list_to_nx_graph(post_pruning_edges, disc_inst.d))
        post_pruning_state.draw_to_file(drawing_file_cam)

    print("=" * 50)
    print("Final results after construction phase:")
    print("=" * 50)
    print_results_to_console(eval_dict['results']['construct'], gt_known)


    if args.include_cam_pruning:
        print("=" * 50)
        print("Final results after CAM pruning phase:")
        print("=" * 50)
        print_results_to_console(eval_dict['results']['prune_cam'], gt_known)


    print(f"Wrote causal discovery results to file {results_file.resolve()}.")

    # pprint.pprint(eval_dict['results']['construct'])
    # pprint.pprint(eval_dict['results']['prune_cam'])

    # perfect_metrics = count_accuracy(disc_inst.true_adj_matrix, disc_inst.true_adj_matrix)
    # print(perfect_metrics)

def print_results_to_console(results_dict, gt_known):
    print(f"Score Function Value: \t\t\t{results_dict['reward']:.3f}")
    print(f"Number of Edges: \t\t\t{results_dict['pred_size']}")
    if gt_known:
        print(f"True Positive Rate (TPR): \t\t{results_dict['tpr']:.3f}")
        print(f"False Discovery Rate (FDR): \t\t{results_dict['fdr']:.3f}")
        print(f"Structural Hamming Distance (SHD): \t{results_dict['shd']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Run causal discovery algorithm for a specified dataset.")
    parser.add_argument("--dataset_file", required=True, type=str,
                        help="Path to dataset file relative to $CD_EXPERIMENT_DATA_DIR. Can be in .csv or .npy format.")

    parser.add_argument("--gt_file", required=False, type=str,
                        help="Path to file containing the adjacency matrix of the ground truth graph relative to $CD_EXPERIMENT_DATA_DIR (optional). Can be in .csv or .npy format.")

    parser.add_argument("--output_directory", required=True, type=str,
                        help="Path to output directory relative to $CD_EXPERIMENT_DATA_DIR.")

    parser.add_argument('--normalize_data', action='store_true', help="Whether to normalize data before running regression.")
    parser.set_defaults(normalize_data=True)

    parser.add_argument("--edge_budget", type=int, required=True, help="Edge budget for the causal discovery problem.")

    parser.add_argument("--reg_type", type=str, required=True, help="Type of regression to run: linear regression (LR), quadratic regression (QR), or Gaussian process regression (GPR).", choices=["LR", "QR", "GPR"])
    parser.add_argument("--score_type", type=str, required=False, help="Score function: BIC with heterogeneous (BIC_different_var) or equal (BIC) variances", choices=["BIC_different_var", "BIC"], default="BIC_different_var")

    parser.add_argument('--include_cam_pruning', action='store_true', help="Whether to apply pruning of edges via statistical tests with the CAM method after the construction phase.")
    parser.set_defaults(include_cam_pruning=False)

    parser.add_argument('--random_seed', type=int, help="Random seed to use as initialization to random number generator.", default=42)

    parser.add_argument("--C_p", type=float, required=False, help="CD-UCT exploration parameter.", default=0.025)
    parser.add_argument('--adjust_C_p', action='store_true', help="Whether to adjust the C_p parameter depending on the Q-value at the root. Recommended to leave on.")
    parser.set_defaults(adjust_C_p=True)

    parser.add_argument("--final_action_strategy", type=str, required=False, help="CD-UCT final mechanism for child action selection .", choices=["max_child", "robust_child"], default="robust_child")
    parser.add_argument("--expansion_budget_modifier", type=float, required=False, help="CD-UCT simulation budget parameter.", default=25)
    parser.add_argument("--sim_policy", type=str, required=False, help="CD-UCT simulation policy.", choices=["random", "naive"], default="random")
    parser.add_argument("--rollout_depth", type=int, required=False, help="CD-UCT rollout depth in terms of number of edges. -1 corresponds to full rollouts (i.e., until the end of the MDP).", default=-1)
    parser.add_argument("--sims_per_expansion", type=int, required=False, help="CD-UCT number of simulations per expansion.", default=1)

    parser.add_argument('--transpositions_enabled', action='store_true', help="Whether to use transpositions when performing the search. Experimental feature. Recommended to leave off.")
    parser.set_defaults(transpositions_enabled=False)

    parser.add_argument('--btm', action='store_true', help="Whether to memorize the best trajectory encountered during the search. Recommended to leave on.")
    parser.set_defaults(btm=True)


    args = parser.parse_args()
    run_causal_discovery(args)


if __name__ == "__main__":
    main()
