from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from cdrl.agent.base_agent import Agent
from cdrl.state.dag_state import DAGState
from cdrl.utils.graph_utils import edge_list_from_adj_matrix, nx_graph_from_adj_matrix, nx_graph_to_adj_matrix


class ExternalAgent(Agent, ABC):
    """
    This type of agent does not operate in the granular MDP environment
    but uses a different formulation of the causal discovery problem.
    """
    def __init__(self, environment):
        super().__init__(environment)

    def make_actions(self, t, **kwargs):
        raise ValueError("This agent is not compatible with the granular MDP. Should be calling overriden eval directly.")

    def eval(self, g_list, phase):
        """
        Evaluates an agent without going through the environment loop.
        Args:
            g_list: list of initial DAGState objects.
            phase: one of [EnvPhase.CONSTRUCT, EnvPhase.PRUNE].

        Returns: The discovered graph structures;
        a list of actions that would correspond to the discovered graphs when ran in the environment;
        and the final objective function values.
        """
        if len(g_list) > 1:
            raise ValueError("not meant to be ran with >1 graph at a time.")

        self.environment.setup(g_list, phase)
        adj_matrix = self.discover_adj_matrix_using_method(self.environment.disc_instance.inputdata)
        print(adj_matrix)

        nx_graph = nx_graph_from_adj_matrix(adj_matrix)
        dag_state = DAGState(nx_graph, init_tracking=False)

        edge_list = edge_list_from_adj_matrix(adj_matrix)
        action_list = [a for e in edge_list for a in e]

        reward = self.environment.get_reward(dag_state)
        return [dag_state], action_list, [reward]

    @abstractmethod
    def discover_adj_matrix_using_method(self, inputdata):
        """
        Base method -- will be implemented differently by each agent.
        Args:
            input_data: Dataset as a pandas DataFrame.
            **kwargs: Keyword arguments.

        Returns: Adjacency matrix of the discovered causal graph as a numpy array.
        """
        pass


class GOBNILPAgent(ExternalAgent):
    """
    An agent based on the GOBNILP algorithm.
    This agent originally interfaced with the algorithm via the `pygobnilp` package.
    However, we later noticed that this library is GPL-licensed. This is not mentioned in its documentation (https://pygobnilp.readthedocs.io/en/latest/)
    or source README (https://bitbucket.org/jamescussens/pygobnilp/src/master/),
    but only on the PyPI page (https://pypi.org/project/pygobnilp/).
    We therefore removed it as dependency and provide the outputs of GOBNILP on the 3 discrete datasets, which are simply read in and reported.
    """
    algorithm_name = 'gobnilp'
    is_deterministic = True
    is_trainable = False


    def discover_adj_matrix_using_method(self, input_data):
        """
        Read and return the appropriate pre-computed graph output by GOBNILP.
        """
        fp = self.options["storage"].file_paths
        instance_name = self.environment.disc_instance.instance_name
        n = self.environment.disc_instance.datasize
        score_type = self.environment.reward_function.score_type

        adj_matrix_file = fp.datasets_storage_dir / "bnlearn" / "processed" / f"{instance_name}-{n}-gobnilp-{score_type}.npy"
        adj_matrix = np.load(str(adj_matrix_file))
        return adj_matrix






