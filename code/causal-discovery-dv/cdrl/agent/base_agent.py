import random
import time
import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import numpy as np

from cdrl.environment.graph_edge_env import EnvPhase
from cdrl.io.file_paths import FilePaths
from cdrl.utils.config_utils import get_logger_instance


class Agent(ABC):
    """
    Abstract base class for all agents.
    """
    def __init__(self, environment):
        """

        Args:
            environment: instance of DirectedGraphEdgeEnv in which the agent operates.
        """
        self.environment = environment
        self.obj_fun_eval_count = 0

    def eval(self, g_list, phase):
        """
        Runs the environment loop. Optionally logs the time taken and number of objective function evaluations performed.
        Args:
            g_list: list of initial DAGState objects.
            phase: one of [EnvPhase.CONSTRUCT, EnvPhase.PRUNE].

        Returns: The final graphs at the end of the construction process;
        the actions taken by the agent;
        and the final rewards received.
        """
        eval_nets = [deepcopy(g) for g in g_list]
        final_graphs = []
        actions = [[] for _ in range(len(g_list))]

        self.environment.setup(eval_nets, phase)
        t = 0
        while not self.environment.is_terminal():

            self.obj_fun_eval_count = 0
            self.log_timings_if_required(t, "before", len(g_list), self.obj_fun_eval_count)
            list_at = self.make_actions(t)

            for i, act in enumerate(list_at):
                actions[i].append(act)

            self.log_timings_if_required(t, "after", len(g_list), self.obj_fun_eval_count)
            self.environment.step(list_at)
            t += 1

        for i in range(len(g_list)):
            final_graphs.append(self.environment.g_list[i].copy())
        rewards = self.environment.rewards

        return final_graphs, actions, rewards

    @abstractmethod
    def make_actions(self, t, **kwargs):
        """
        Core method of the agent that implements the decision-making algorithm.
        Args:
            t: Current environment timestep.
            **kwargs: Keyword arguments.

        Returns: list of actions (one for each graph in the current environment).

        """
        pass

    def setup(self, options, hyperparams):
        """
        Configures an agent with the given options and hyperparameters.
        Args:
            options: dictionary whose values control various aspects of agent behaviour (not algorithm-related).
            hyperparams: hyperparameters of the algorithm.

        """
        self.options = options
        if 'log_timings' in options:
            self.log_timings = options['log_timings']
        else:
            self.log_timings = False

        if self.log_timings:
            self.setup_timings_file()


        if 'log_filename' in options:
            self.log_filename = options['log_filename']
        else:
            self.log_filename = None

        if 'log_progress' in options:
            self.log_progress = options['log_progress']
        else:
            self.log_progress = False
        if self.log_progress:
            self.logger = get_logger_instance(self.log_filename)
            self.environment.pass_logger_instance(self.logger)
        else:
            self.logger = None

        if 'random_seed' in options:
            self.set_random_seeds(options['random_seed'])
        else:
            self.set_random_seeds(42)
        self.hyperparams = hyperparams

    def get_default_hyperparameters(self):
        return {}

    def setup_timings_file(self):
        """
        Creates a file for tracking agent wall clock times and number of objective function calls.
        """
        self.timings_path = self.options['timings_path']
        timings_filename = self.timings_path / FilePaths.construct_timings_file_name(self.options['model_identifier_prefix'])
        timings_file = Path(timings_filename)
        if timings_file.exists():
            timings_file.unlink()
        self.timings_out = open(timings_filename, 'a')

    def log_timings_if_required(self, t, entry_tag, num_graphs, obj_fun_eval_count):
        """
        Writes wall clock times and number of objective function calls, if required.
        """
        if self.log_timings and self.timings_out is not None:
            ms_since_epoch = time.time() * 1000
            self.timings_out.write('%d,%s,%d,%.6f,%d\n' % (t, entry_tag, num_graphs, ms_since_epoch, obj_fun_eval_count))
            try:
                self.timings_out.flush()
            except BaseException:
                if self.logger is not None:
                    self.logger.warn("caught an exception when trying to flush timings data.")
                    self.logger.warn(traceback.format_exc())

    def finalize(self):
        """
        Cleans up resources after agent is no longer needed.
        """
        if self.log_timings:
            if self.timings_out is not None and not self.timings_out.closed:
                self.timings_out.close()

    def pick_random_actions(self, i):
        """
        Chooses uniformly at random among valid edges in the graph.
        Args:
            i: index of the graph under consideration out of the environment's g_list.

        Returns: tuple corresponding to chosen directed edge.

        """
        g = self.environment.g_list[i]
        if self.environment.phase == EnvPhase.CONSTRUCT:
            banned_first_nodes = g.banned_actions

            first_valid_acts = self.environment.get_valid_actions(g, banned_first_nodes)
            if len(first_valid_acts) == 0:
                return -1, -1

            first_node = self.local_random.choice(tuple(first_valid_acts))
            banned_second_nodes = g.get_invalid_edge_ends(self.environment.phase, first_node, enforce_acyclic=self.environment.enforce_acyclic)
            second_valid_acts = self.environment.get_valid_actions(g, banned_second_nodes)

            second_node = self.local_random.choice(tuple(second_valid_acts))

            return first_node, second_node
        else:
            existing_edges = g.get_edge_list()
            rand_edge = self.local_random.choice(existing_edges)
            return rand_edge[0], rand_edge[1]


    def set_random_seeds(self, random_seed):
        """
        Configures the local random number generator and sets the seed for the used libraries.
        Args:
            random_seed: integer random seed.

        """
        self.random_seed = random_seed
        self.local_random = random.Random()
        self.local_random.seed(self.random_seed)
        np.random.seed(self.random_seed)
