from copy import deepcopy
from enum import Enum

import numpy as np

class EnvPhase(Enum):
    """
    Enumeration that determines whether the environment is constructing (i.e., adding edges) or pruning (i.e., removing edges).
    Only the CONSTRUCT phase was used in the experiments; PRUNE is also supported but did not lead to consistent improvements
    over pruning using statistical tests with CAM.
    """
    CONSTRUCT = 1
    PRUNE = 2

class DirectedGraphEdgeEnv(object):
    """
    Class that implements the environment mechanics that decision-making agents interact with.
    """
    def __init__(self, disc_instance, reward_function, initial_edge_budgets, enforce_acyclic, **kwargs):
        self.disc_instance = disc_instance
        self.reward_function = reward_function
        self.initial_edge_budgets = initial_edge_budgets

        self.enforce_acyclic = enforce_acyclic

        self.num_mdp_substeps = 2
        self.g_list = None


    def setup(self, g_list, phase):
        """
        Sets up the environment for use by a decision-making agent.
        Args:
            g_list: list of initial DAGState object on which the environment operates.
            As agents decide modifications to the graph structure, these are applied to the g_list and the environment advances to the next step.
            The environment supports operating on several graphs at the same time, but in this repository it is typically used with only 1 graph at a time.

            phase: EnvPhase
        """
        self.g_list = g_list
        self.n_steps = 0

        self.phase = phase
        budget = self.initial_edge_budgets[str(phase.name.lower())]

        self.edge_budgets = np.array([budget] * len(self.g_list))
        self.used_edge_budgets = np.zeros(len(g_list), dtype=np.float)
        self.exhausted_budgets = np.zeros(len(g_list), dtype=np.bool)

        for i in range(len(self.g_list)):
            g = g_list[i]
            g.first_node = None
            g.populate_banned_actions(self.phase, budget=self.edge_budgets[i], enforce_acyclic=self.enforce_acyclic)

        self.rewards = np.zeros(len(g_list), dtype=np.float)


    def pass_logger_instance(self, logger):
        """Pass logger instance for use in the environment."""
        self.logger_instance = logger

    def get_reward(self, graph):
        """Calculate the reward for the given graph."""
        return self.reward_function.calculate_reward_single_graph(graph)

    def get_rewards(self, graphs):
        """Calculate rewards for a list of graphs."""
        return np.array([self.get_reward(g) for g in graphs])

    def get_graph_edge_choices_for_idx(self, i):
        """Return the choice of possible edges for the graph at the ith index in the g_list."""
        g = self.g_list[i]
        return self.get_graph_edge_choices(g, self.phase, self.enforce_acyclic)

    @staticmethod
    def get_graph_edge_choices(g, phase, enforce_acyclic):
        """Return the valid edge choices (i.e., candidate edges) for the given graph."""
        if phase == EnvPhase.CONSTRUCT:
            banned_first_nodes = g.banned_actions
            valid_acts = DirectedGraphEdgeEnv.get_valid_actions(g, banned_first_nodes)
            non_edges = set()
            for first in valid_acts:
                banned_second_nodes = g.get_invalid_edge_ends(phase, first, enforce_acyclic=enforce_acyclic)
                valid_second_nodes = DirectedGraphEdgeEnv.get_valid_actions(g, banned_second_nodes)

                for second in valid_second_nodes:
                    non_edges.add((first, second))
            return non_edges
        else:
            existing_edges = g.get_edge_list()
            return existing_edges


    def get_remaining_budget(self, i):
        """Returns the remaining edge budget for the graph at the ith index in the g_list."""
        return self.edge_budgets[i] - self.used_edge_budgets[i]

    @staticmethod
    def get_valid_actions(g, banned_actions):
        """Returns the valid node choices given forbidden ones."""
        all_nodes_set = g.all_nodes_set
        valid_first_nodes = all_nodes_set - banned_actions
        return valid_first_nodes

    @staticmethod
    def apply_action(g, phase, action, remaining_budget, enforce_acyclic, copy_state=False):
        """Applies the chosen action by modifying the graph structure of the state."""
        if g.first_node is None:
            if copy_state:
                g_ref = g.copy()
            else:
                g_ref = g
            g_ref.first_node = action
            g_ref.populate_banned_actions(phase, budget=remaining_budget, enforce_acyclic=enforce_acyclic)
            # selection doesn't cost anything.
            return g_ref, remaining_budget
        else:
            modify_fn = g.add_edge if phase == EnvPhase.CONSTRUCT else g.remove_edge
            new_g, edge_cost = modify_fn(g.first_node, action)
            new_g.first_node = None

            updated_budget = remaining_budget - edge_cost
            new_g.populate_banned_actions(phase, budget=updated_budget, enforce_acyclic=enforce_acyclic)
            return new_g, updated_budget

    @staticmethod
    def apply_action_in_place(g, phase, action, remaining_budget, enforce_acyclic):
        """Applies the chosen action but does not modify the actual structure of the graph, instead "storing" the modification in the state.
        This is used extensively in the simulation policy of Monte Carlo Tree Search, where we are only interested in the final state and not intermediary ones.
         """
        if g.first_node is None:
            g.first_node = action
            g.populate_banned_actions(phase, budget=remaining_budget, enforce_acyclic=enforce_acyclic)
            return remaining_budget
        else:
            modify_fn = g.add_edge_dynamically if phase == EnvPhase.CONSTRUCT else g.remove_edge_dynamically
            edge_cost = modify_fn(g.first_node, action)
            g.first_node = None

            updated_budget = remaining_budget - edge_cost
            g.populate_banned_actions(phase, budget=updated_budget, enforce_acyclic=enforce_acyclic)
            return updated_budget

    def step(self, actions):
        """Applies the decided actions for each graph in the g_list, updating edge budgets and termination condition accordingly."""
        for i in range(len(self.g_list)):
            if not self.exhausted_budgets[i]:
                if actions[i] == -1:
                    if self.logger_instance is not None:
                        self.logger_instance.warn("budget not exhausted but trying to apply dummy action!")
                        self.logger_instance.error(f"the remaining budget: {self.get_remaining_budget(i)}")
                        g = self.g_list[i]

                        self.logger_instance.error(f"first_node selection: {g.first_node}")


                remaining_budget = self.get_remaining_budget(i)
                self.g_list[i], updated_budget = self.apply_action(self.g_list[i], self.phase, actions[i], remaining_budget, self.enforce_acyclic)
                # print(f"graph index {i}: updated budget from {remaining_budget} to {updated_budget}.")

                self.used_edge_budgets[i] += (remaining_budget - updated_budget)

                # after action applied, if step uneven, check if banned nodes is all nodes
                # if it is, then update is_terminal status and populate entry in reward array
                if self.n_steps % 2 == 1:
                    if self.g_list[i].banned_actions == self.g_list[i].all_nodes_set:
                        self.mark_exhausted(i)

        self.n_steps += 1

    def mark_exhausted(self, i):
        """Marks the budget as exhausted for the ith graph in the g_list."""
        self.exhausted_budgets[i] = True
        reward = self.get_reward(self.g_list[i])
        self.rewards[i] = reward


    def is_terminal(self):
        """If budget is exhausted for all graphs, environment execution can stop."""
        return np.all(self.exhausted_budgets)

