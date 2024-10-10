from copy import copy, deepcopy
from random import Random

import math
import networkx as nx
import numpy as np

from cdrl.environment.graph_edge_env import DirectedGraphEdgeEnv, EnvPhase
from cdrl.state.dag_state import get_graph_hash
from cdrl.utils.graph_utils import contains_cycles_exact


class SimulationPolicy(object):
    """
    Base class for simulation policies.
    """
    def __init__(self, local_random, **kwargs):
        self.local_random = local_random

        self.enforce_acyclic = kwargs.pop("enforce_acyclic")
        self.max_parents = kwargs.pop("max_parents")


    def get_random_state(self):
        return self.local_random.getstate()

class RandomSimulationPolicy(SimulationPolicy):
    """
    Uniform random simulation policy.
    """
    def __init__(self,  local_random, **kwargs):
        super().__init__(local_random, **kwargs)

    def choose_action(self, state, possible_actions, current_depth):
        available_acts = tuple(possible_actions)
        chosen_action = self.local_random.choice(available_acts)
        return chosen_action

    def reset_for_new_simulation(self, start_state):
        pass