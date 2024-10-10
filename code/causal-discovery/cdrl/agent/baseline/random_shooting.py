from cdrl.agent.base_agent import Agent
from cdrl.agent.mcts.simulation_policies import RandomSimulationPolicy
from cdrl.utils.graph_utils import contains_cycles


class RandomShootingAgent(Agent):
    """
    Baseline agent referred to as "Random Search" in the reported results.
    This agent carries out a number of simulations in the MDP and memorizes the trajectory corresponding to the best encountered graph.
    """
    is_trainable = False
    algorithm_name = 'randomshooting'
    is_deterministic = False


    def __init__(self, environment):
        super().__init__(environment)

    def create_sim_policy(self):
        """
        Initializes the random simulation policy (same one as used by vanilla CD-UCT).
        """
        sim_policy_class = RandomSimulationPolicy
        self.sim_policy_inst = sim_policy_class(self.local_random, **{})


    def make_actions(self, t, **kwargs):
        raise ValueError("method not supported -- should call eval using overriden method")

    def eval(self, g_list, phase):
        """
        Evaluates the RandomShooting agent. A separate implementation of eval is required due to
        the fact that many simulations are carried out; the environment is cloned and reset on each simulation.

        Args:
            g_list: list of initial DAGState objects.
            phase: one of [EnvPhase.CONSTRUCT, EnvPhase.PRUNE].

        Returns: The discovered graph structures;
        a list of actions that would correspond to the discovered graphs when ran in the environment;
        and the final objective function values.
        """
        starting_graph = g_list[0]

        self.environment.setup([starting_graph], phase)
        edge_budget = self.environment.edge_budgets[0]
        # compute a simulation budget equivalent to the one provided to CD-UCT.
        sim_budget = int(starting_graph.num_nodes * self.hyperparams['expansion_budget_modifier'] * edge_budget * 2)

        best_graph, best_acts, best_reward = None, None, float("-inf")

        self.create_sim_policy()

        for j in range(sim_budget):
            if j % 1000 == 0:
                print(f"running sim {j}/{sim_budget}.")

            self.environment.setup([starting_graph], phase)
            actions = []
            state = starting_graph.copy()
            state.init_dynamic_edges()

            current_depth = 0
            rem_budget = self.environment.get_remaining_budget(0)

            self.sim_policy_inst.reset_for_new_simulation(starting_graph)

            while True:
                possible_actions = self.environment.get_valid_actions(state, state.banned_actions)
                if rem_budget <= 0 or len(possible_actions) == 0:
                    break

                available_acts = tuple(possible_actions)
                chosen_action = self.sim_policy_inst.choose_action(state, available_acts, current_depth)
                actions.append(chosen_action)
                rem_budget = self.environment.apply_action_in_place(state, phase, chosen_action, rem_budget, self.environment.enforce_acyclic)
                current_depth += 1

            final_graph = state.apply_dynamic_edges(phase)
            reward = self.environment.get_reward(final_graph)

            discovered_adj = final_graph.get_adjacency_matrix()
            assert (not contains_cycles(discovered_adj))


            if reward > best_reward:
                best_graph = final_graph
                best_acts = actions
                best_reward = reward
                print(f"updated best reward to {best_reward}")


        return [best_graph], [best_acts], [best_reward]