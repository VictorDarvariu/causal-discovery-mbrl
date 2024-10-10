import gc
from copy import deepcopy
from pathlib import Path

import math
from math import sqrt, log

from tqdm import tqdm

from cdrl.agent.base_agent import Agent
from cdrl.agent.mcts.mcts_tree_node import MCTSTreeNode
from cdrl.agent.mcts.simulation_policies import RandomSimulationPolicy
from cdrl.environment.graph_edge_env import EnvPhase
from cdrl.evaluation.eval_utils import *
from cdrl.state.dag_state import budget_eps, get_graph_hash
from cdrl.utils.general_utils import get_memory_usage_str, read_file_as_np_array


class MonteCarloTreeSearchAgent(Agent):
    """
    Implementation of Monte Carlo Tree Search as described in
    Browne, C. B., Powley, E., Whitehouse, D., Lucas, S. M., Cowling, P. I., Rohlfshagen, P., ... & Colton, S. (2012). A survey of monte carlo tree search methods.
    IEEE Transactions on Computational Intelligence and AI in games, 4(1), 1-43.
    """
    algorithm_name = "uct"

    is_deterministic = False
    is_trainable = False

    def __init__(self, environment):
        super().__init__(environment)
        self.root_nodes = None

        self.final_action_strategies = {'max_child': self.pick_max_child,
                                        'robust_child': self.pick_robust_child,
                                        }

    def init_root_information(self, t, phase):
        """
        Initializes the root of the search tree and the bookkeeping functionality.
        """
        self.root_nodes = []
        self.node_expansion_budgets = []

        if self.transpositions_enabled:
            self.tree_indices = []
            self.tree_indices_hit_count = []


        for i in range(len(self.environment.g_list)):
            if self.transpositions_enabled:
                self.tree_indices.append(dict())
                self.tree_indices_hit_count.append(0)

            start_state = self.environment.g_list[i]
            remaining_budget = self.environment.get_remaining_budget(i)

            root_node = self.initialize_tree_node(i, None, start_state, None, remaining_budget, phase, with_depth=t)
            exp_budget = int(start_state.num_nodes * self.expansion_budget_modifier)

            self.root_nodes.append(root_node)
            self.node_expansion_budgets.append(exp_budget)


    def make_actions(self, t, **kwargs):
        raise ValueError("method not supported -- should call eval using overriden method")


    def eval(self, g_list, phase):
        """
        Custom evaluation loop implemented as the agent (optionally) memorizes the best trajectory encountered across all simulations.
        Args:
            g_list: list of initial DAGState objects.
            phase: one of [EnvPhase.CONSTRUCT, EnvPhase.PRUNE].

        Returns: The final graphs at the end of the construction process;
        the actions taken by the agent;
        and the final rewards received.
        """
        if len(g_list) > 1:
            raise ValueError("not meant to be ran with >1 graph at a time.")

        self.environment.setup(g_list, phase)
        trajectories = [self.run_trajectory_collection()]
        graphs = [t[0] for t in trajectories]
        acts = [t[1] for t in trajectories]
        rewards = [t[2] for t in trajectories]
        return graphs, acts, rewards

    def run_trajectory_collection(self):
        """Main loop of the algorithm that runs MCTS for each timestep of the environment."""
        acts = []

        t = 0
        max_steps = int(2 * self.environment.get_remaining_budget(0))

        with tqdm(total=max_steps, colour="green") as pbar:
            pbar.set_description(f"CD-UCT Algorithm Main Loop")

            while not self.environment.is_terminal():
                self.obj_fun_eval_count = 0
                self.log_timings_if_required(t, "before", 1, self.obj_fun_eval_count)
                self.run_search_for_g_list(self.environment.phase, t, force_init=True)
                list_at = self.pick_children()
                self.log_timings_if_required(t, "after", 1, self.obj_fun_eval_count)

                acts.append(list_at[0])
                self.environment.step(list_at)
                t += 1
                pbar.update(1)


            if self.btm:
                final_graph = self.best_graphs_found[0]
                best_acts = self.best_trajectories_found[0]
                best_F = self.best_Fs[0]
            else:
                final_graph = self.environment.g_list[0].copy()
                best_acts = self.moves_so_far[0]
                best_F = self.environment.rewards[0]

                if math.isnan(best_F):
                    best_F = 0.


        return final_graph, best_acts, best_F

    def run_search_for_g_list(self, phase, t, force_init=True):
        """
        Carries out MCTS for a single timestep for all graphs in the environment g_list.
        Args:
            phase: one of [EnvPhase.CONSTRUCT, EnvPhase.PRUNE].
            t: the environment timestep.
            force_init: Whether the tree is recreated from scratch at each timestep. Setting this to False keeps some of the tree nodes
            created at the previous timestep, but introduces a bias in the search.

        """
        if t == 0:
            self.moves_so_far = []

            self.best_graphs_found = []
            self.best_trajectories_found = []
            self.best_Fs = []

            self.trajectory_data = []

            self.C_ps = []
            self.starting_budgets = []
            self.rollout_limits = []

            self.create_sim_policy()

            for i in range(len(self.environment.g_list)):
                g = self.environment.g_list[i]

                self.moves_so_far.append([])

                self.best_graphs_found.append(None)
                self.best_trajectories_found.append([])
                self.best_Fs.append(float("-inf"))


                starting_budget = self.environment.get_remaining_budget(i)
                self.starting_budgets.append(starting_budget)

                if self.hyperparams['rollout_depth'] == -1:
                    self.rollout_limits.append(starting_budget)
                else:
                    rollout_limit = min(starting_budget, self.hyperparams['rollout_depth'])
                    self.rollout_limits.append(rollout_limit)
                self.C_ps.append(self.hyperparams['C_p'])


            if self.hyperparams['adjust_C_p']:
                self.init_root_information(t, phase)
                for i in range(len(self.root_nodes)):
                    self.execute_search_step(i, t, phase)

        if t == 0 or force_init:
            self.init_root_information(t, phase)
        for i in range(len(self.root_nodes)):
            self.execute_search_step(i, t, phase)

    def execute_search_step(self, i, t, phase):
        """Carries out MCTS for a single timestep for a single graph in the environment g_list."""
        if self.log_progress:
            self.logger.info(f"executing step {t}")
            self.logger.info(f"{get_memory_usage_str()}")

        root_node = self.root_nodes[i]
        node_expansion_budget = self.node_expansion_budgets[i]

        starting_budget = self.starting_budgets[i]
        rollout_limit = self.rollout_limits[i]

        num_simulations = 0

        while True:
            # follow tree policy to reach a certain node
            tree_nodes, tree_actions = self.follow_tree_policy(root_node, i, phase)

            v_l = tree_nodes[-1]
            simulation_results = self.execute_simulation_policy(v_l, root_node, i, starting_budget, rollout_limit, t)
            num_simulations += len(simulation_results)
            self.obj_fun_eval_count += len(simulation_results)

            if self.transpositions_enabled:
                for _, _, R, _, _ in simulation_results:
                    self.update_estimates_recursive(v_l, R)
            else:
                self.backup_values(tree_nodes, tree_actions, simulation_results)

            if self.btm:
                self.update_best_trajectories(i, t, tree_actions, simulation_results)

            if num_simulations >= node_expansion_budget:
                root_Q = root_node.Q
                if self.hyperparams['adjust_C_p']:
                    # This optionally adjusts the value of the C_p parameter to achieve consistent exploration.
                    # This accounts for the fact that the values of states encountered at later timesteps of the search are higher than those encountered early on.
                    if root_Q < 0:
                        # prevents a negative confidence interval in the UCB computation.
                        self.C_ps[i] = self.hyperparams['C_p'] * (-root_Q)
                    else:
                        self.C_ps[i] = self.hyperparams['C_p'] * (root_Q)

                break


    def create_sim_policy(self):
        """Intializes the simulation policy class."""
        if self.sim_policy == "random":
            self.sim_policy_inst = RandomSimulationPolicy(self.local_random, enforce_acyclic=self.environment.enforce_acyclic, max_parents=self.environment.max_parents)
        else:
            raise ValueError(f"sim policy {self.sim_policy} not recognised!")

    def update_estimates_recursive(self, current_node, R):
        """Runs the backpropagation procedure recursively to update nodes towards the root in the case transpositions are used
        and multiple parents are possible per node."""
        current_node.update_estimates(R)

        if current_node.parent_nodes is None:
            return

        for parent_node in current_node.parent_nodes:
            self.update_estimates_recursive(parent_node, R)

    def pick_children(self):
        """When a MCTS step is completed, chooses the child node of the root as the next action."""
        actions = []
        for i in range(len(self.root_nodes)):
            root_node = self.root_nodes[i]
            if len(root_node.children) > 0:
                action, selected_child = self.final_action_strategy(root_node)
                self.root_nodes[i] = selected_child
                self.root_nodes[i].parent_nodes = None
                del root_node
                gc.collect()
            else:
                self.environment.mark_exhausted(i)
                action = -1

            self.moves_so_far[i].append(action)
            actions.append(action)
        return actions

    def pick_robust_child(self, root_node):
        """Chooses the child with the most visits as next action."""
        return sorted(root_node.children.items(), key=lambda x: (x[1].N, x[1].Q), reverse=True)[0]

    def pick_max_child(self, root_node):
        """Chooses the child with the highest average reward as next action."""
        return sorted(root_node.children.items(), key=lambda x: (x[1].Q, x[1].N), reverse=True)[0]

    def follow_tree_policy(self, node, i, phase):
        """Follows the tree policy by applying the UCB rule until a node needs to be created and added to the tree."""
        traversed_nodes = []
        actions_taken = []

        if node.num_valid_actions == 0:
            traversed_nodes.append(node)
            return traversed_nodes, actions_taken

        while True:
            traversed_nodes.append(node)
            state = node.state

            if len(node.children) < node.num_valid_actions:
                if hasattr(self, 'step'):
                    global_step = self.step
                else:
                    global_step = 1

                chosen_action = node.choose_action(int(self.random_seed * node.N * global_step))
                next_state, updated_budget = self.environment.apply_action(state, phase, chosen_action, node.remaining_budget,
                                                                           self.environment.enforce_acyclic, self.environment.max_parents,copy_state=True)


                if self.transpositions_enabled:
                    next_state_hash = get_graph_hash(next_state, size=64, include_first=True)

                    if next_state_hash in self.tree_indices[i]:
                        # We came across this state on another path in the tree -- join up the paths.
                        next_node = self.tree_indices[i][next_state_hash]
                        next_node.parent_nodes.append(node)
                        self.tree_indices_hit_count[i] += 1

                    else:
                        next_node = self.initialize_tree_node(i, [node], next_state, chosen_action, updated_budget, phase)
                        self.tree_indices[i][next_state_hash] = next_node

                else:
                    next_node = self.initialize_tree_node(i, [node], next_state, chosen_action, updated_budget, phase)

                node.children[chosen_action] = next_node
                actions_taken.append(chosen_action)
                traversed_nodes.append(next_node)

                break
            else:
                if node.num_valid_actions == 0:
                    break
                else:
                    highest_ucb, ucb_action, ucb_node = self.pick_best_child(node, i, self.C_ps[i])
                    node = ucb_node
                    actions_taken.append(ucb_action)
                    continue

        return traversed_nodes, actions_taken

    def initialize_tree_node(self, i, parent_nodes, node_state, chosen_action, updated_budget, phase, with_depth=-1):
        """Creates a new node in the search tree."""
        banned_actions = node_state.banned_actions
        next_node_actions = self.environment.get_valid_actions(node_state, banned_actions)

        next_node_actions = list(next_node_actions)

        parent_depth = parent_nodes[0].depth if parent_nodes is not None else 0
        depth = parent_depth + 1 if with_depth == -1 else with_depth

        predictor_vals = self.get_predictor_values(node_state, next_node_actions)
        next_node = MCTSTreeNode(node_state, parent_nodes, chosen_action, next_node_actions, remaining_budget=updated_budget, depth=depth)
        next_node.assign_predictor_values(predictor_vals)

        if self.transpositions_enabled:
            state_hash = get_graph_hash(node_state, size=64, include_first=True)
            self.tree_indices[i][state_hash] = next_node

        return next_node

    def get_predictor_values(self, state, actions):
        """Returns uniform (i.e., all ones) predictor values for all possible actions.
        Non-uniform predictor values are useful for AlphaGo-like MCTS agents that use a neural network predictor to bias the search; but this is not used here."""
        n = len(actions)
        if n == 0:
            return []
        uniform_probs = np.full(n, 1, dtype=np.float32)
        return uniform_probs


    def pick_best_child(self, node, i, c):
        """Chooses best child by the given node selection criterion."""
        highest_value = float("-inf")
        best_node = None
        best_action = None

        child_values = {action: self.node_selection_strategy(node, i, action, child_node)
                        for action, child_node in node.children.items()}


        for action, value in child_values.items():
            if value > highest_value:
                highest_value = value
                best_node = node.children[action]
                best_action = action
        return highest_value, best_action, best_node

    def node_selection_strategy(self, parent_node, i, action, child_node):
        """Node selection is based on UCB1 for default UCT."""
        Q, parent_N, child_N = child_node.Q, parent_node.N, child_node.N
        predictor_value = parent_node.get_predictor_value(action)

        node_value = self.get_ucb1_term(Q, parent_N, child_N, self.C_ps[i], predictor_value)

        if math.isnan(node_value):
            node_value = 0.0

        return node_value

    def get_ucb1_term(self, Q, parent_N, child_N, C_p, model_prior):
        """Computes the UCB1 value based on the supplied node statistics."""
        ci_term = sqrt((2 * log(parent_N)) / child_N)
        ucb1_value = Q + C_p * model_prior * ci_term
        return ucb1_value

    def execute_simulation_policy(self, node, root_node, i, starting_budget, rollout_limit, t):
        """Executes the simulation policy upon the creation of a new tree node."""
        if self.hyperparams['rollout_depth'] == 0:
            return [(node.state, [], self.get_final_node_val(node.state, self.environment.reward_function), None, 0)]

        if node.num_valid_actions == 0:
            return [(node.state, [], self.get_final_node_val(node.state, self.environment.reward_function), None, 0)]

        valid_actions_finder = self.environment.get_valid_actions

        action_applier = self.environment.apply_action_in_place
        simulation_results = []


        for sim_number in range(self.sims_per_expansion):
            final_state, out_of_tree_acts, R, post_random_state, sim_number = self.sim_policy_episode(self.sim_policy_inst,
                                                                                                node,
                                                                                                root_node,
                                                                                                starting_budget,
                                                                                                rollout_limit,
                                                                                                sim_number,
                                                                                                valid_actions_finder,
                                                                                                action_applier,
                                                                                                self.environment.reward_function,
                                                                                                self.environment.enforce_acyclic,
                                                                                                self.environment.max_parents,
                                                                                                self.local_random.getstate(),
                                                                                                t,
                                                                                                self.environment.phase
                                                                                                )
            simulation_results.append((final_state, out_of_tree_acts, R, post_random_state, sim_number))
            self.local_random.setstate(post_random_state)
        return simulation_results

    @staticmethod
    def sim_policy_episode(sim_policy,
                                  node,
                                  root_node,
                                  starting_budget,
                                  rollout_limit,
                                  sim_number,
                                  valid_actions_finder,
                                  action_applier,
                                  reward_function,
                                  enforce_acyclic,
                                  max_parents,
                                  random_state,
                                  t,
                                  phase
                                  ):
        """Executes the simulation policy by sampling actions using the specified strategy until a terminal state is reached.
        This and corresponding sub-methods were made static to enable parallel simulations using a thread pool, but this feature was not used in the end.
        """

        initial_depth, rem_budget = MonteCarloTreeSearchAgent.find_budget_at_leaf(root_node, rollout_limit, node)

        state = node.state.copy()
        state.init_dynamic_edges()

        out_of_tree_actions = []
        current_rollout_depth = 0

        sim_policy.reset_for_new_simulation(state)

        while True:
            possible_actions = valid_actions_finder(state, state.banned_actions)
            total_depth = initial_depth + current_rollout_depth

            if rem_budget <= budget_eps or len(possible_actions) == 0:
                break

            available_acts = tuple(possible_actions)
            chosen_action = sim_policy.choose_action(state, available_acts, total_depth)
            out_of_tree_actions.append(chosen_action)

            rem_budget = action_applier(state, phase, chosen_action, rem_budget, enforce_acyclic, max_parents)

            current_rollout_depth += 1

        final_state = state.apply_dynamic_edges(phase)

        node_val = MonteCarloTreeSearchAgent.get_final_node_val(final_state, reward_function)
        post_random_state = sim_policy.get_random_state()
        return final_state, out_of_tree_actions, node_val, post_random_state, sim_number

    @staticmethod
    def find_budget_at_leaf(root_node, rollout_limit, node):
        """
        Computes the edge budget available for a leaf node from which a simulation is about to start.
        Required as the search may attempt to go further forward in time than the remaining budget at the root.
        """
        initial_depth = node.depth
        used_so_far = root_node.remaining_budget - node.remaining_budget
        rem_budget = rollout_limit - used_so_far
        rem_budget = min(node.remaining_budget, rem_budget)
        rem_budget = max(0, rem_budget)
        return initial_depth, rem_budget

    @staticmethod
    def get_final_node_val(final_state, reward_function):
        """Value of a terminal node is the reward function as evaluated at the final state."""
        final_value = reward_function.calculate_reward_single_graph(final_state)
        return final_value

    def update_best_trajectories(self, i, t, tree_actions, simulation_results):
        """Updates the best encountered trajectories during the course of the search."""
        for final_state, out_of_tree_acts, F, _, sim_num in simulation_results:
            phase = self.environment.phase
            if phase == EnvPhase.CONSTRUCT:
                should_update = True
            elif phase == EnvPhase.PRUNE:
                should_update = (final_state.num_edges + (self.starting_budgets[i])) == (self.root_nodes[i].state.num_edges + math.floor(self.root_nodes[i].depth / 2))

            else:
                raise ValueError(f"invalid phase {phase}")

            if should_update:
                if F > self.best_Fs[i]:
                    self.best_Fs[i] = F

                    best_traj = deepcopy(self.moves_so_far[i])
                    best_traj.extend(tree_actions)
                    best_traj.extend(out_of_tree_acts)

                    self.best_trajectories_found[i] = best_traj
                    self.best_graphs_found[i] = final_state
                    if self.log_progress:
                        self.logger.info(f"updated best reward to {F}")


    def backup_values(self, tree_nodes, tree_actions, simulation_results):
        """Executes backup for the given simulation results, updating the estimates of all tree nodes along the path from the root."""
        for tree_node in tree_nodes:
            for _, _, R, _, _ in simulation_results:
                tree_node.update_estimates(R)


    def setup(self, options, hyperparams):
        """
        Configures an agent with the given options and hyperparameters.
        Args:
            options: dictionary whose values control various aspects of agent behaviour (not algorithm-related).
            hyperparams: hyperparameters of the algorithm.
        """
        super().setup(options, hyperparams)

        # determines the number of simulations that are ran when a new tree node is created (default 1, used throughout).
        if 'sims_per_expansion' in hyperparams:
            self.sims_per_expansion = hyperparams['sims_per_expansion']
        else:
            self.sims_per_expansion = 1

        # exploration parameter of the algorithm.
        self.C_p = hyperparams['C_p']

        # parameter that determines the number of expansions performed by the algorithm.
        # the number of expansions is set to the number of nodes times this parameter, to account for larger graphs typically requiring more simulations for accurate estimates.
        self.expansion_budget_modifier = hyperparams['expansion_budget_modifier']

        # whether to memorize the best trajectory encounted during the search. can lead to substantial improvements.
        self.btm = hyperparams['btm']
        if self.log_progress:
            self.logger.info(f">>> setting btm to <<{self.btm}>>.")

        # whether to join up nodes that are encountered along different paths in the search tree.
        # this is an experimental feature and was not used in the reported experiments.
        self.transpositions_enabled = hyperparams['transpositions_enabled']


        # simulation policy to use.
        if 'sim_policy' in hyperparams:
            self.sim_policy = hyperparams['sim_policy']
        else:
            self.sim_policy = 'random'

        # depth of the rollout (possibly truncated)
        if 'rollout_depth' in hyperparams:
            self.rollout_depth = hyperparams['rollout_depth']
        else:
            self.rollout_depth = -1

        # how the child node of the root is chosen as the next action once the tree search step is completed.
        if 'final_action_strategy' in hyperparams:
            self.final_action_strategy = self.final_action_strategies[hyperparams['final_action_strategy']]
        else:
            self.final_action_strategy = self.pick_robust_child


    def get_default_hyperparameters(self):
        """Specifies sensible default hyperparameters in case they are not provided."""
        default_hyperparams = {
            'C_p': 0.02,
            'adjust_C_p': True,
            'final_action_strategy': 'max_child',
            'expansion_budget_modifier': 25,
            'sim_policy': 'random',
            'rollout_depth': -1,
            'sims_per_expansion': 1,
            'transpositions_enabled': False,
            'btm': True
        }
        return default_hyperparams

    def finalize(self):
        """Tidies up the agent before deallocation. """
        self.root_nodes = None
        self.node_expansion_budgets = None


# The following subclasses operate in the same manner, and differ only in the choice of some parameters.
# This makes it easier to configure and track the results for several classes of agents.
class UCTDepth2Agent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uctd2'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['rollout_depth'] = 2
        super().setup(options, hyps_copy)


class UCTDepth4Agent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uctd4'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['rollout_depth'] = 4
        super().setup(options, hyps_copy)


class UCTDepth8Agent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uctd8'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['rollout_depth'] = 8
        super().setup(options, hyps_copy)


class UCTDepth16Agent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uctd16'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['rollout_depth'] = 16
        super().setup(options, hyps_copy)


class UCTDepth32Agent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uctd32'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['rollout_depth'] = 32
        super().setup(options, hyps_copy)


class UCTFullDepthAgent(MonteCarloTreeSearchAgent):
    algorithm_name = 'uctfull'

    def setup(self, options, hyperparams):
        hyps_copy = deepcopy(hyperparams)
        hyps_copy['rollout_depth'] = -1
        super().setup(options, hyps_copy)