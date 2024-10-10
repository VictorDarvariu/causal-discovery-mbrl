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
        adj_matrix = self.discover_adj_matrix_using_method(self.environment.disc_instance.inputdata, rvar_type=self.environment.disc_instance.rvar_type)
        print(adj_matrix)

        nx_graph = nx_graph_from_adj_matrix(adj_matrix)
        dag_state = DAGState(nx_graph, init_tracking=False)

        edge_list = edge_list_from_adj_matrix(adj_matrix)
        action_list = [a for e in edge_list for a in e]

        reward = self.environment.get_reward(dag_state)
        return [dag_state], action_list, [reward]

    @abstractmethod
    def discover_adj_matrix_using_method(self, input_data, **kwargs):
        """
        Base method -- will be implemented differently by each agent.
        Args:
            input_data: Dataset as a two-dimensional numpy array.
            **kwargs: Keyword arguments.

        Returns: Adjacency matrix of the discovered causal graph as a numpy array.
        """
        pass

class CAMAgent(ExternalAgent):
    """
    Agent based on the CAM algorithm.
    Note that this agent only performs the "construction" phase of CAM.
    Pruning is performed later using the same routine as RL-BIC and CD-UCT so that
    the techniques are comparable.

    Uses the implementation in the cdt package.
    """
    algorithm_name = 'cam'
    is_deterministic = True
    is_trainable = False

    def discover_adj_matrix_using_method(self, input_data, **kwargs):
        from cdt.causality.graph import CAM
        cam_obj = CAM(variablesel=True, cutoff=0.001, pruning=False)
        out_nx = cam_obj.predict(pd.DataFrame(input_data))
        adj_matrix = nx_graph_to_adj_matrix(out_nx)
        return adj_matrix


class GESAgent(ExternalAgent):
    """
    Agent based on the GES algorithm.
    Uses the implementation in the cdt package.
    """
    algorithm_name = 'ges'
    is_deterministic = True
    is_trainable = False

    def discover_adj_matrix_using_method(self, input_data, **kwargs):
        from cdt.causality.graph import GES

        rvar_type = kwargs.pop('rvar_type', None)
        if rvar_type == "discrete":
            which_score = "int"
        elif rvar_type == "continuous":
            which_score = "obs"
        else:
            raise ValueError(f"invalid {rvar_type}")

        ges_obj = GES(score=which_score, verbose=True)
        out_nx = ges_obj.predict(pd.DataFrame(input_data))
        adj_matrix = nx_graph_to_adj_matrix(out_nx)
        return adj_matrix

class PCAgent(ExternalAgent):
    """
    Agent based on the PC algorithm.
    Uses the implementation in the cdt package.
    """
    algorithm_name = 'pc'
    is_deterministic = True
    is_trainable = False

    def discover_adj_matrix_using_method(self, input_data, **kwargs):
        from cdt.causality.graph import PC

        rvar_type = kwargs.pop('rvar_type', None)
        if rvar_type == "discrete":
            which_test = "discrete"
        elif rvar_type == "continuous":
            which_test = "gaussian"
        else:
            raise ValueError(f"invalid {rvar_type}")

        pc_obj = PC(CItest=which_test)
        out_nx = pc_obj.predict(pd.DataFrame(input_data))
        adj_matrix = nx_graph_to_adj_matrix(out_nx)
        return adj_matrix



class LiNGAMAgent(ExternalAgent):
    """
    Agent based on the LiNGAM algorithm.
    Uses the implementation provided by the orignal authors.
    """
    algorithm_name = 'lingam'
    is_deterministic = True
    is_trainable = False

    def __init__(self, environment):
        super().__init__(environment)
        import lingam
        self.model = lingam.DirectLiNGAM()

    def discover_adj_matrix_using_method(self, input_data, **kwargs):
        self.model.fit(input_data)
        adj_matrix = self.model.adjacency_matrix_
        adj_matrix[adj_matrix != 0.] = 1.
        return adj_matrix


class NOTEARSAgent(ExternalAgent):
    """
    Agent based on the NOTEARS algorithm.
    Uses the implementation provided by the orignal authors.
    """
    algorithm_name = 'notears'
    is_deterministic = True
    is_trainable = False

    def discover_adj_matrix_using_method(self, input_data, **kwargs):
        from notears.linear import notears_linear
        adj_matrix = notears_linear(input_data, lambda1=self.hyperparams['lambda1'],
                                    w_threshold=self.hyperparams['w_threshold'], loss_type='l2')

        adj_matrix[adj_matrix != 0.] = 1.
        return adj_matrix

    def get_default_hyperparameters(self):
        return {"lambda1": 0.1, "w_threshold": 0.3}





