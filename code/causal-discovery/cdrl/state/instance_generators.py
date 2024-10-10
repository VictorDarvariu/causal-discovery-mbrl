import random
from collections import namedtuple
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from cdrl.state.dag_state import DAGState
from cdrl.utils.graph_utils import nx_graph_from_adj_matrix

InstanceMetadata = namedtuple("InstanceMetadata", ['name', 'rvar_type', 'transpose', 'root_path', 'reg_type', 'rlbic_num_edges'])

class HardcodedInstanceGenerator(object):
    """
    Generates causal discovery instance from the known continuous variable datasets.
    """
    name = "hardcoded"

    KNOWN_INSTANCES = {
        "sachs": InstanceMetadata(name="sachs", rvar_type="continuous", transpose=False, root_path="/experiment_data/datasets/sachs", reg_type="GPR", rlbic_num_edges=49),

        "syntren1": InstanceMetadata(name="syntren1", rvar_type="continuous", transpose=False, root_path="/experiment_data/datasets/syntren/1", reg_type="GPR", rlbic_num_edges=97),
        "syntren2": InstanceMetadata(name="syntren2", rvar_type="continuous", transpose=False, root_path="/experiment_data/datasets/syntren/2", reg_type="GPR", rlbic_num_edges=97),
        "syntren3": InstanceMetadata(name="syntren3", rvar_type="continuous", transpose=False, root_path="/experiment_data/datasets/syntren/3", reg_type="GPR", rlbic_num_edges=97),
        "syntren4": InstanceMetadata(name="syntren4", rvar_type="continuous", transpose=False, root_path="/experiment_data/datasets/syntren/4", reg_type="GPR", rlbic_num_edges=97),
        "syntren5": InstanceMetadata(name="syntren5", rvar_type="continuous", transpose=False, root_path="/experiment_data/datasets/syntren/5", reg_type="GPR", rlbic_num_edges=97),
        "syntren6": InstanceMetadata(name="syntren6", rvar_type="continuous", transpose=False, root_path="/experiment_data/datasets/syntren/6", reg_type="GPR", rlbic_num_edges=97),
        "syntren7": InstanceMetadata(name="syntren7", rvar_type="continuous", transpose=False, root_path="/experiment_data/datasets/syntren/7", reg_type="GPR", rlbic_num_edges=97),
        "syntren8": InstanceMetadata(name="syntren8", rvar_type="continuous", transpose=False, root_path="/experiment_data/datasets/syntren/8", reg_type="GPR", rlbic_num_edges=97),
        "syntren9": InstanceMetadata(name="syntren9", rvar_type="continuous", transpose=False, root_path="/experiment_data/datasets/syntren/9", reg_type="GPR", rlbic_num_edges=97),
        "syntren10": InstanceMetadata(name="syntren10", rvar_type="continuous", transpose=False, root_path="/experiment_data/datasets/syntren/10", reg_type="GPR", rlbic_num_edges=97),

        "synth50qr": InstanceMetadata(name="synth50qr", rvar_type="continuous", transpose=True, root_path="/experiment_data/datasets/synthetic/50nodesqr/gauss_same_noise/1", reg_type="QR", rlbic_num_edges=-1),

        "synth10lr": InstanceMetadata(name="synth10lr", rvar_type="continuous", transpose=True, root_path="/experiment_data/datasets/synthetic/10nodeslr/gauss_same_noise/1", reg_type="LR", rlbic_num_edges=-1),
        "synth15lr": InstanceMetadata(name="synth15lr", rvar_type="continuous", transpose=True, root_path="/experiment_data/datasets/synthetic/15nodeslr/gauss_same_noise/1", reg_type="LR", rlbic_num_edges=-1),
        "synth20lr": InstanceMetadata(name="synth20lr", rvar_type="continuous", transpose=True, root_path="/experiment_data/datasets/synthetic/20nodeslr/gauss_same_noise/1", reg_type="LR", rlbic_num_edges=-1),
        "synth25lr": InstanceMetadata(name="synth25lr", rvar_type="continuous", transpose=True, root_path="/experiment_data/datasets/synthetic/25nodeslr/gauss_same_noise/1", reg_type="LR", rlbic_num_edges=-1),
        "synth30lr": InstanceMetadata(name="synth30lr", rvar_type="continuous", transpose=True, root_path="/experiment_data/datasets/synthetic/30nodeslr/gauss_same_noise/1", reg_type="LR", rlbic_num_edges=-1),
        "synth35lr": InstanceMetadata(name="synth35lr", rvar_type="continuous", transpose=True, root_path="/experiment_data/datasets/synthetic/35nodeslr/gauss_same_noise/1", reg_type="LR", rlbic_num_edges=-1),
        "synth40lr": InstanceMetadata(name="synth40lr", rvar_type="continuous", transpose=True, root_path="/experiment_data/datasets/synthetic/40nodeslr/gauss_same_noise/1", reg_type="LR", rlbic_num_edges=-1),
        "synth45lr": InstanceMetadata(name="synth45lr", rvar_type="continuous", transpose=True, root_path="/experiment_data/datasets/synthetic/45nodeslr/gauss_same_noise/1", reg_type="LR", rlbic_num_edges=-1),
        "synth50lr": InstanceMetadata(name="synth50lr", rvar_type="continuous", transpose=True, root_path="/experiment_data/datasets/synthetic/50nodeslr/gauss_same_noise/1", reg_type="LR", rlbic_num_edges=-1),
    }

    @staticmethod
    def get_instance(**kwargs):
        instance_name = kwargs.pop("instance_name")
        metadata = HardcodedInstanceGenerator.KNOWN_INSTANCES[instance_name]
        data_path = '{}/data.npy'.format(metadata.root_path)
        dag_path = '{}/DAG.npy'.format(metadata.root_path)

        return DiscoveryInstance(metadata, data_path=data_path, dag_path=dag_path, **kwargs)

class SynthGPInstanceGenerator(object):
    """
    Generates causal discovery instance from the known synthetic datasets generated using Gaussian Process Regression.
    """
    name = "synthgp"
    metadata = InstanceMetadata(name="synthgp", rvar_type="continuous", transpose=False, root_path="/experiment_data/datasets/gpgen", reg_type="GPR", rlbic_num_edges=-1)

    @staticmethod
    def get_instance(**kwargs):
        root_dir = Path(SynthGPInstanceGenerator.metadata.root_path)
        gt = kwargs.pop("gt") # topology: "er"
        n = kwargs.pop("n") # number of samples in training set
        p = kwargs.pop("p") # number of variables in graph
        e = kwargs.pop("e") # approximate number of edges desired in graph
        what_vary = kwargs.pop("what_vary")  # what to vary: graph density or amount of data.
        model_seed = kwargs.pop("model_seed")

        if what_vary == "density":
            dag_index = int(model_seed / 42) + 1
            datadir = root_dir / f"data_{gt}_p{p}_e{e}_n{n}_GP"
            data_path, dag_path = datadir / f"data{dag_index}.npy", datadir / f"DAG{dag_index}.npy"
            print(f"loading dag at {dag_path}")
            return DiscoveryInstance(SynthGPInstanceGenerator.metadata, data_path=data_path, dag_path=dag_path, subsample_data=False, **kwargs)
        else:
            dag_index = 1 # all seeds get the same DAG but the dataset is subsampled.
            largest_dataset = 10000
            datadir = root_dir / f"data_{gt}_p{p}_e{e}_n{largest_dataset}_GP"
            data_path, dag_path = datadir / f"data{dag_index}.npy", datadir / f"DAG{dag_index}.npy"
            print(f"loading dag at {dag_path}")
            return DiscoveryInstance(SynthGPInstanceGenerator.metadata, data_path=data_path, dag_path=dag_path, subsample_data=True, subsample_n=n, subsample_seed=model_seed, **kwargs)


class DiscoveryInstance(object):
    """
    Instance of the causal discovery problem.
    """
    def __init__(self, instance_metadata, data_path=None, dag_path=None, normalize_data=True, starting_graph_generation="scratch", subsample_data=False, **kwargs):
        super().__init__()
        self.instance_metadata = instance_metadata

        self.instance_name = instance_metadata.name
        self.instance_path = instance_metadata.root_path
        self.rvar_type = instance_metadata.rvar_type

        self.data = self.read_file_as_np_array(data_path)

        if subsample_data:
            subsample_n = kwargs.pop("subsample_n")
            subsample_seed = kwargs.pop("subsample_seed")

            self.local_random = random.Random()
            self.local_random.seed(subsample_seed)
            self.data = resample(self.data, n_samples=subsample_n, replace=False, random_state=self.local_random.randint(0, (2 ** 32) - 1))

        self.datasize, self.d = self.data.shape

        if normalize_data:
            self.inputdata = StandardScaler().fit_transform(self.data)

        if dag_path is not None:
            gtrue = self.read_file_as_np_array(dag_path)

            if self.instance_metadata.transpose:
                gtrue = np.transpose(gtrue)

            # (i,j)=1 => node i -> node j
            self.true_adj_matrix = np.int32(np.abs(gtrue) > 1e-3)
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
            raise ValueError(f"graph generation mechanism {starting_graph_generation} not supported.")


    def read_file_as_np_array(self, file_path):
        file_pathlib = Path(file_path)
        if file_pathlib.suffix == ".npy":
            np_arr = np.load(file_path)
        elif file_pathlib.suffix == ".csv":
            np_arr = pd.read_csv(file_path, header=None).values
        else:
            raise ValueError(f"Unsupported data file type {file_pathlib.suffix}.")

        return np_arr



