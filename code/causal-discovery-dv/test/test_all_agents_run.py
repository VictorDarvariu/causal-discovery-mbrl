import pytest

from cdrl.agent.baseline.baseline_agent import *
from cdrl.agent.baseline.random_shooting import RandomShootingAgent
from cdrl.agent.mcts.mcts_agent import *
from cdrl.environment.graph_edge_env import DirectedGraphEdgeEnv, EnvPhase
from cdrl.evaluation.eval_utils import get_metrics_dict
from cdrl.io.file_paths import FilePaths
from cdrl.io.storage import EvaluationStorage
from cdrl.reward_functions.reward_discrete_vars import DiscreteVarsBICRewardFunction
from cdrl.state.instance_generators import BNLearnInstanceGenerator


@pytest.mark.parametrize(
    "agent_class",
    [
    UCTDepth2Agent,
    UCTDepth4Agent,
    UCTDepth8Agent,
    UCTDepth16Agent,
    UCTDepth32Agent,
    UCTFullDepthAgent,

    GreedyAgent,
    RandomAgent,
    RandomShootingAgent,
])
def test_agent_runs(agent_class):
    inst_name = "asia"
    score_type = "BIC"

    n = 1000
    max_parents = None

    disc_inst = BNLearnInstanceGenerator.get_instance(instance_name=inst_name, n=n)
    rfun = DiscreteVarsBICRewardFunction(disc_inst, score_type=score_type)

    initial_edge_budgets = {
        "construct": disc_inst.true_num_edges,
        "prune": 0
    }

    env = DirectedGraphEdgeEnv(disc_inst, rfun, initial_edge_budgets=initial_edge_budgets, enforce_acyclic=True, max_parents=max_parents)
    agent = agent_class(env)

    hyps = agent.get_default_hyperparameters()
    hyps['expansion_budget_modifier'] = 1

    opts = {"random_seed": 42,
            "log_progress": True,
            "storage": EvaluationStorage(FilePaths("/experiment_data", "development", setup_directories=False)),
    }

    agent.setup(opts, hyps)
    agent.eval([disc_inst.start_state], EnvPhase.CONSTRUCT)
