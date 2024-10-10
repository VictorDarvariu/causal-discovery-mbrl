from copy import deepcopy
from pathlib import Path

from cdrl.agent.baseline.baseline_agent import GreedyAgent, RandomAgent
from cdrl.agent.baseline.random_shooting import RandomShootingAgent
from cdrl.agent.mcts.mcts_agent import UCTFullDepthAgent, UCTDepth8Agent, UCTDepth4Agent, UCTDepth2Agent, UCTDepth16Agent, UCTDepth32Agent, NaiveUCTFullDepthAgent
from cdrl.agent.baseline.external_agent import CAMAgent, LiNGAMAgent, NOTEARSAgent, GESAgent, PCAgent
from cdrl.agent.baseline.rlbic_agent import RLBICAgent

from cdrl.reward_functions.reward_continuous_vars import ContinuousVarsBICRewardFunction
from cdrl.state.instance_generators import HardcodedInstanceGenerator, SynthGPInstanceGenerator

known_cmd_kwargs = ["gt", "n", "p", "e", "what_vary"]
class ExperimentConditions(object):
    """
    This class and associated subclasses specify the conditions for an experiment.
    This is done in Python code as some parameter and option choices need to be set programatically.
    """
    def __init__(self, instance_name, vars_dict):
        self.instance_name = instance_name

        for k, v in vars_dict.items():
            if k in known_cmd_kwargs:
                setattr(self, k, v)

        self.perform_construction = True
        self.perform_pruning = False
        self.starting_graph_generation = "scratch"

        self.btm_on_eval = True

        self.objective_functions = [
            ContinuousVarsBICRewardFunction
        ]

        self.network_generators = [
            HardcodedInstanceGenerator,
        ]

        self.hyps_chunk_size = 1
        self.seeds_chunk_size = 1


        self.agents = self.get_agents()

    def get_agents(self):
        return [
            self.get_mcts_class(self.instance_name),
            RLBICAgent,

            GreedyAgent,
            RandomShootingAgent,
            RandomAgent,

            CAMAgent,
            LiNGAMAgent,
            NOTEARSAgent,

            GESAgent,
            PCAgent,
        ]

    def get_mcts_class(self, instance_name):
        if instance_name == "sachs":
            return UCTFullDepthAgent
        elif instance_name.startswith("syntren"):
            return UCTDepth8Agent
        elif instance_name == "synth50qr":
            return UCTDepth4Agent
        else:
            return UCTFullDepthAgent


class MainExperimentConditions(ExperimentConditions):
    def __init__(self, instance_name, vars_dict):
        super().__init__(instance_name, vars_dict)

        self.experiment_params = {'num_runs': 50, # number of random seeds
                                  # 'score_type': "BIC", # BIC with equal variances
                                  'score_type': "BIC_different_var", # BIC with heterogeneous variances
                                  'enforce_acyclic': True,
                                  'penalise_cyclic': True,
                                  }

        self.num_seeds_skip = 0
        self.seed_offset = int(self.num_seeds_skip * 42)

        self.test_seeds = [self.seed_offset + int(run_number * 42) for run_number in range(self.experiment_params['num_runs'])]
        self.validation_seeds = [self.seed_offset + int(run_number * 42) for run_number in range(len(self.test_seeds), self.experiment_params['num_runs'] + len(self.test_seeds))]

        self.hyperparam_grids = self.create_hyperparam_grids(vars_dict)

    def create_hyperparam_grids(self, vars_dict):
        if "budget" in vars_dict:
            budget_modifiers = [vars_dict["budget"]]
        else:
            budget_modifiers = [25]

        base_mcts = {
            "C_p": [0.025, 0.05, 0.075, 0.1],
            "expansion_budget_modifier": budget_modifiers,
            "sims_per_expansion": [1],
            "adjust_C_p": [True],
            "sim_policy": ["random"],
            "final_action_strategy": ["robust_child"],
            "transpositions_enabled": [False],
        }
        hyperparam_grid_base = {
            UCTDepth2Agent.algorithm_name: deepcopy(base_mcts),
            UCTDepth4Agent.algorithm_name: deepcopy(base_mcts),
            UCTDepth8Agent.algorithm_name: deepcopy(base_mcts),
            UCTDepth16Agent.algorithm_name: deepcopy(base_mcts),
            UCTDepth32Agent.algorithm_name: deepcopy(base_mcts),
            UCTFullDepthAgent.algorithm_name: deepcopy(base_mcts),

            RLBICAgent.algorithm_name: {"nb_epoch": [20000],
                                        "input_dimension": [32, 64],
                                        "lr1_start": [0.001, 0.0001]
                                        },

            RandomShootingAgent.algorithm_name: {
                "expansion_budget_modifier": budget_modifiers,
            },

            NOTEARSAgent.algorithm_name: {"lambda1": [0.1], "w_threshold": [0.3]},

        }

        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)

        return hyperparam_grids

    def get_initial_edge_budgets(self, network_generator, discovery_instance):
        if network_generator == HardcodedInstanceGenerator and discovery_instance.instance_metadata.rlbic_num_edges != -1:
            return {
                "construct": discovery_instance.instance_metadata.rlbic_num_edges,
                "prune": 0
            }
        else:
            return {
                "construct": discovery_instance.true_num_edges,
                "prune": 0
            }

class BudgetExperimentConditions(MainExperimentConditions):
    def __init__(self, instance_name, vars_dict):
        super().__init__(instance_name, vars_dict)

    def get_agents(self):
        # only MCTS and Random Search have varying budgets for this experiment.
        return [
            self.get_mcts_class(self.instance_name),
            RandomShootingAgent,
        ]

class TimingsExperimentConditions(MainExperimentConditions):
    def __init__(self, instance_name, vars_dict):
        super().__init__(instance_name, vars_dict)

    def get_agents(self):
        return [
            UCTFullDepthAgent,
            NaiveUCTFullDepthAgent,
        ]
    def create_hyperparam_grids(self, vars_dict):
        base_mcts = {
            "C_p": [0.025],
            "expansion_budget_modifier": [vars_dict["budget"]],
            "sims_per_expansion": [1],
            "adjust_C_p": [True],
            "sim_policy": ["random"],
            "final_action_strategy": ["robust_child"],
            "transpositions_enabled": [False],
        }
        hyperparam_grid_base = {
            UCTFullDepthAgent.algorithm_name: deepcopy(base_mcts),
            NaiveUCTFullDepthAgent.algorithm_name: deepcopy(base_mcts),
        }

        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)

        return hyperparam_grids

class ScaleupExperimentConditions(MainExperimentConditions):
    def __init__(self, instance_name, vars_dict):
        super().__init__(instance_name, vars_dict)

    def get_agents(self):
        all_agents = super().get_agents()
        all_agents.remove(RLBICAgent) # excluded due to lack of scalability to n=50 graphs.
        return all_agents

    def create_hyperparam_grids(self, vars_dict):
        if "budget" in vars_dict:
            budget_modifiers = [vars_dict["budget"]]
        else:
            budget_modifiers = [25]

        base_mcts = {
            "C_p": [0.025],
            "expansion_budget_modifier": budget_modifiers,
            "sims_per_expansion": [1],
            "adjust_C_p": [True],
            "sim_policy": ["random"],
            "final_action_strategy": ["robust_child"],
            "transpositions_enabled": [False],
        }
        hyperparam_grid_base = {
            UCTDepth4Agent.algorithm_name: deepcopy(base_mcts),
            RandomShootingAgent.algorithm_name: {
                "expansion_budget_modifier": budget_modifiers,
            },
            NOTEARSAgent.algorithm_name: {"lambda1": [0.1], "w_threshold": [0.3]},
        }

        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)

        return hyperparam_grids

class SyntheticExperimentConditions(MainExperimentConditions):
    def __init__(self, instance_name, vars_dict):
        super().__init__(instance_name, vars_dict)
        self.network_generators = [
            SynthGPInstanceGenerator
        ]

    def get_agents(self):
        return [
            self.get_mcts_class(self.instance_name),
            RandomShootingAgent,
            GreedyAgent,
            RandomAgent
        ]

    def create_hyperparam_grids(self, vars_dict):
        if "budget" in vars_dict:
            budget_modifiers = [vars_dict["budget"]]
        else:
            budget_modifiers = [25]

        base_mcts = {
            "C_p": [0.1],
            "expansion_budget_modifier": budget_modifiers,
            "sims_per_expansion": [1],
            "adjust_C_p": [True],
            "sim_policy": ["random"],
            "final_action_strategy": ["robust_child"],
            "transpositions_enabled": [False],
        }
        hyperparam_grid_base = {
            UCTFullDepthAgent.algorithm_name: deepcopy(base_mcts),
            RandomShootingAgent.algorithm_name: {
                "expansion_budget_modifier": budget_modifiers,
            },
        }

        hyperparam_grids = {}
        for f in self.objective_functions:
            hyperparam_grids[f.name] = deepcopy(hyperparam_grid_base)

        return hyperparam_grids


def get_conditions_for_experiment(which, instance_name, cmd_args):
    if hasattr(cmd_args, "__dict__"):
        vars_dict = vars(cmd_args)
    else:
        vars_dict = cmd_args

    if which == 'main':
        cond = MainExperimentConditions(instance_name, vars_dict)
    elif which == 'budget':
        cond = BudgetExperimentConditions(instance_name, vars_dict)
    elif which == 'timings':
        cond = TimingsExperimentConditions(instance_name, vars_dict)
    elif which == 'scaleup':
        cond = ScaleupExperimentConditions(instance_name, vars_dict)
    elif which == 'synthetic':
        cond = SyntheticExperimentConditions(instance_name, vars_dict)
    else:
        raise ValueError(f"experiment {which} not recognized!")
    return cond
