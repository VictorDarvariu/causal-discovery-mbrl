from cdrl.agent.baseline.baseline_agent import *
from cdrl.agent.baseline.external_agent import GOBNILPAgent
from cdrl.agent.baseline.random_shooting import RandomShootingAgent
from cdrl.agent.mcts.mcts_agent import *

from cdrl.reward_functions.reward_discrete_vars import DiscreteVarsBICRewardFunction
from cdrl.state.instance_generators import BNLearnInstanceGenerator


class ExperimentConditions(object):
    """
    This class and associated subclasses specify the conditions for an experiment.
    This is done in Python code as some parameter and option choices need to be set programatically.
    """
    def __init__(self, instance_name, vars_dict):
        self.instance_name = instance_name
        self.n = 1000 # note: currently hardcoded

        self.perform_construction = True
        self.perform_pruning = False

        self.starting_graph_generation = "scratch"
        self.btm_on_hyperopt = True
        self.btm_on_eval = True

        self.objective_functions = [
            DiscreteVarsBICRewardFunction
        ]

        self.network_generators = [
            BNLearnInstanceGenerator
        ]

        self.hyps_chunk_size = 1
        self.seeds_chunk_size = 1

    def set_params_and_seeds(self):
        self.experiment_params = {'num_runs': self.num_runs, # number of random seeds
                                  'score_type': "BIC",
                                  'enforce_acyclic': True,
                                  'penalise_cyclic': True,

                                  'max_parents': None,
                                  'store_scores': False,
                                  }

        self.num_seeds_skip = 0
        self.seed_offset = int(self.num_seeds_skip * 42)

        self.test_seeds = [self.seed_offset + int(run_number * 42) for run_number in range(self.experiment_params['num_runs'])]
        self.validation_seeds = [self.seed_offset + int(run_number * 42) for run_number in range(len(self.test_seeds), self.experiment_params['num_runs'] + len(self.test_seeds))]


    def get_initial_edge_budgets(self, network_generator, discovery_instance):
        return {
            "construct": discovery_instance.true_num_edges,
            "prune": 0
        }


class MainExperimentConditions(ExperimentConditions):
    def __init__(self, instance_name, vars_dict):
        super().__init__(instance_name, vars_dict)

        self.num_runs = 50
        self.set_params_and_seeds()

        self.agents = [
            UCTDepth2Agent,
            GreedyAgent,
            RandomAgent,
            RandomShootingAgent,
            GOBNILPAgent,
        ]


        self.hyperparam_grids = self.create_hyperparam_grids(vars_dict)

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
            "final_action_strategy": ["max_child"],
            "transpositions_enabled": [False],
        }

        hyperparam_grid_base = {
            UCTDepth2Agent.algorithm_name: deepcopy(base_mcts),
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
    else:
        raise ValueError(f"experiment {which} not recognized!")
    return cond
