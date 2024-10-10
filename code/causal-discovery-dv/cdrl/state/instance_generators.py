from collections import namedtuple

import networkx as nx
import numpy as np
import pandas as pd

from cdrl.state.dag_state import DAGState
from cdrl.utils.general_utils import read_file_as_np_array
from cdrl.utils.graph_utils import nx_graph_from_adj_matrix

InstanceMetadata = namedtuple("InstanceMetadata", ['name', 'rvar_type', 'transpose', 'root_path', 'reg_type', 'rlbic_num_edges'])

class BNLearnInstanceGenerator(object):
    """
    Generates causal discovery instance from the known BNLearn datasets.
    """
    name = "bnlearn"

    KNOWN_INSTANCES = {
        "asia": InstanceMetadata(name="asia", rvar_type="discrete", transpose=False, root_path="/experiment_data/datasets/bnlearn/processed", reg_type=None, rlbic_num_edges=None),
        "child": InstanceMetadata(name="child", rvar_type="discrete", transpose=False, root_path="/experiment_data/datasets/bnlearn/processed", reg_type=None, rlbic_num_edges=None),
        "insurance": InstanceMetadata(name="insurance", rvar_type="discrete", transpose=False, root_path="/experiment_data/datasets/bnlearn/processed", reg_type=None, rlbic_num_edges=None),
    }

    @staticmethod
    def get_instance(**kwargs):
        instance_name = kwargs.pop("instance_name")
        metadata = BNLearnInstanceGenerator.KNOWN_INSTANCES[instance_name]
        n = kwargs.get("n", 1000)  # number of samples in training set -- default 1000.

        dag_path = '{}/{}-{}-gt.npy'.format(metadata.root_path, instance_name, n)
        data_path = '{}/{}-{}.csv'.format(metadata.root_path, instance_name, n)
        return DiscoveryInstance(metadata, data_path=data_path, dag_path=dag_path, **kwargs)


class DiscoveryInstance(object):
    """
    Instance of the causal discovery problem.
    """
    def __init__(self, instance_metadata, data_path=None, dag_path=None, starting_graph_generation="scratch", **kwargs):
        super().__init__()
        self.instance_metadata = instance_metadata

        self.instance_name = instance_metadata.name
        self.instance_path = instance_metadata.root_path
        self.rvar_type = instance_metadata.rvar_type

        self.data = pd.read_csv(data_path)

        self.node_aliases = list(self.data.columns)
        self.inputdata = self.data

        self.datasize, self.d = self.data.shape

        if dag_path is not None:
            self.true_adj_matrix = read_file_as_np_array(dag_path)
            self.true_num_edges = np.count_nonzero(self.true_adj_matrix)
            self.true_graph = DAGState(nx_graph_from_adj_matrix(self.true_adj_matrix), init_tracking=False)

        else:
            self.true_adj_matrix = None
            self.true_num_edges = None
            self.true_graph = None

        # currently only start from scratch, but can easily be modified to start from a given graph structure.
        if starting_graph_generation == "scratch":
            empty_dag = nx.DiGraph()
            empty_dag.add_nodes_from(list(range(self.d)))
            self.start_state = DAGState(empty_dag)
        else:
            raise ValueError(f"graph generation {starting_graph_generation} not supported.")

