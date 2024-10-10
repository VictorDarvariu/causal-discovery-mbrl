import subprocess
import numpy as np
from cdrl.agent.baseline.external_agent import ExternalAgent


class RLBICAgent(ExternalAgent):
    """
    Interface to the RL-BIC agent whose source code is included under the rlbic-src/ directory.
    We opt for calling it externally since it is easier to interface with the main script provided by the authors.
    """
    algorithm_name = 'rlbic'
    is_deterministic = False
    is_trainable = False

    WHICH_PYTHON = "/opt/conda/envs/cd-env/bin/python"
    RL_BIC_MAIN = f"/causal-discovery/rlbic-src/main.py"

    def __init__(self, environment):
        super().__init__(environment)


    def discover_adj_matrix_using_method(self, input_data, **kwargs):
        disc_inst = self.environment.disc_instance

        command = [self.WHICH_PYTHON, self.RL_BIC_MAIN,
                   "--max_length", str(disc_inst.d),
                   "--data_size", str(disc_inst.datasize),
                   "--score_type", self.environment.reward_function.score_type,
                   "--reg_type", disc_inst.instance_metadata.reg_type,
                   "--read_data",
                   "--data_path", disc_inst.instance_metadata.root_path,
                   "--lambda_flag_default",
                   "--nb_epoch", str(self.hyperparams["nb_epoch"]),
                   "--input_dimension", str(self.hyperparams["input_dimension"]),
                   "--lr1_start", str(self.hyperparams["lr1_start"]),
                   "--lambda_iter_num", str(1000),
                   "--seed", str(self.random_seed)
                   ]

        if disc_inst.instance_metadata.transpose:
            command.extend(["--transpose"])
        else:
            command.extend(["--use_bias", "--bias_initial_value", str(-10), "--normalize"])

        if disc_inst.instance_metadata.name == "synth50qr":
            command.extend(["--use_bias", "--bias_initial_value", str(-10)])


        storage = self.options['storage']
        output_dir = storage.file_paths.rlbic_output_dir / self.options['model_identifier_prefix']
        output_dir.mkdir(exist_ok=True, parents=True)

        output_dir_str = str(output_dir.absolute())

        command.append("--output_dir")
        command.append(output_dir_str)

        retcode = subprocess.run(command).returncode
        if retcode == 0:
            dag_path = output_dir / "final_graphs" / "DAG_orig.npy"
            if dag_path.exists():
                adj_matrix = np.load(str(dag_path.absolute()))
            else:
                # RL-BIC was run for too few steps to produce the output; return no edges.
                return np.zeros((disc_inst.d, disc_inst.d), dtype=np.float32)

            return adj_matrix
        else:
            raise Exception(f"Failed to run RL-BIC agent. Exited with code {retcode}.")

    def get_default_hyperparameters(self):
        return {"nb_epoch": 20000,
                "input_dimension": 128,
                "lr1_start": 0.001}



