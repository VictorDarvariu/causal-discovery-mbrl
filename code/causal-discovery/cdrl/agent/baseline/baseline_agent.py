from abc import abstractmethod, ABC

from cdrl.agent.base_agent import Agent
from cdrl.environment.graph_edge_env import EnvPhase


class BaselineAgent(Agent, ABC):
    """
    Base class for several baseline agents that operate within the directed graph construction environment.
    """
    is_trainable = False

    def __init__(self, environment):
        """

        Args:
            environment: an instance of DirectedGraphEdgeEnv in which the agent operates.
        """
        super().__init__(environment)
        self.future_actions = None

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)

    def make_actions(self, t, **kwargs):
        """
        Several baseline agents make decisions on an edge-by-edge basis, but the MDP definition
        implemented by the environment allows choosing a single node per timestep.
        Upon choosing the edges, their "starting points" are returned immediately;
        while the "endpoints" are stored in the future_actions attribute and will be returned on the next timestep.

        Args:
            t: the current timestep.
            **kwargs:

        Returns: a list of actions corresponding to the chosen node for each graph in the environment's g_list.

        """
        if t % 2 == 0:
            first_actions, second_actions = [], []
            for i in range(len(self.environment.g_list)):
                first_node, second_node = self.pick_actions_using_strategy(t, i)
                first_actions.append(first_node)
                second_actions.append(second_node)

            self.future_actions = second_actions
            chosen_actions = first_actions

        else:
            chosen_actions = self.future_actions
            self.future_actions = None

        return chosen_actions

    def finalize(self):
        pass

    @abstractmethod
    def pick_actions_using_strategy(self, t, i):
        pass


class RandomAgent(BaselineAgent):
    """
    Baseline agent that chooses nodes uniformly at random.
    """
    algorithm_name = 'random'
    is_deterministic = False

    def __init__(self, environment):
        super().__init__(environment)

    def pick_actions_using_strategy(self, t, i):
        return self.pick_random_actions(i)


class GreedyAgent(BaselineAgent):
    """
    Baseline agent that chooses the edge that improves the objective function most.
    """
    algorithm_name = 'greedy'
    is_deterministic = True

    def __init__(self, environment):
        super().__init__(environment)

    def pick_actions_using_strategy(self, t, i):
        if self.log_progress:
            self.logger.info(f"{self.algorithm_name} executing greedy step {t}")
        first_node, second_node = None, None

        g = self.environment.g_list[i]
        edge_choices = list(self.environment.get_graph_edge_choices_for_idx(i))
        if len(edge_choices) == 0:
            return (-1, -1)
        elif len(edge_choices) == 1:
            return edge_choices[0][0], edge_choices[0][1]

        initial_value = self.environment.rewards[i]
        best_val = float("-inf")

        for first, second in edge_choices:
            g_copy = g.copy()

            modify_fn = g_copy.add_edge if self.environment.phase == EnvPhase.CONSTRUCT else g_copy.remove_edge
            next_g, _ = modify_fn(first, second)

            edge_val = self.get_edge_value(initial_value, g, next_g)
            self.obj_fun_eval_count += 1

            if edge_val > best_val:
                best_val = edge_val
                first_node, second_node = first, second

        return first_node, second_node


    def get_edge_value(self, initial_value, g, next_g):
        next_value = self.environment.get_reward(next_g)
        return next_value - initial_value