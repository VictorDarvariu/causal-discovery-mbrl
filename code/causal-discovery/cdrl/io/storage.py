import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from cdrl.io.file_paths import FilePaths


class EvaluationStorage:
    """
    Class for storing and retrieving hyperparameter optimisation and evaluation data.
    """
    EXPERIMENT_DETAILS_FILENAME = "experiment_details.json"

    def __init__(self, file_paths):
        self.file_paths = file_paths

    def get_hyperparameter_optimisation_data(self,
                                     experiment_id,
                                     model_seeds_to_skip,
                                     train_individually):
        """Retrieve and summarise all raw hyperparameter optimisation data from a given experiment."""
        latest_experiment = self.get_experiment_details(experiment_id)
        file_paths = latest_experiment["file_paths"]
        experiment_conditions = latest_experiment["experiment_conditions"]

        hyperopt_data = []

        network_generators = latest_experiment["network_generators"]
        objective_functions = latest_experiment["objective_functions"]
        agent_names = latest_experiment["agents"]
        param_spaces = latest_experiment["parameter_search_spaces"]

        for objective_function in objective_functions:
            for agent_name in agent_names:

                if agent_name in param_spaces[objective_function]:
                    agent_grid = param_spaces[objective_function][agent_name]
                    search_space_keys = list(agent_grid.keys())

                    for hyperparams_id in search_space_keys:
                        for seed in experiment_conditions['validation_seeds']:
                            for network_generator in network_generators:

                                graph_id = None

                                setting = (network_generator, objective_function, agent_name, graph_id)
                                if setting in model_seeds_to_skip:
                                    if seed in model_seeds_to_skip[setting]:
                                        print(f"Skipping seed {seed} when computing optimal hyperparams.")
                                        continue

                                model_prefix = FilePaths.construct_model_identifier_prefix(agent_name,
                                                                                       objective_function,
                                                                                       network_generator,
                                                                                       seed,
                                                                                       hyperparams_id,
                                                                                       graph_id=graph_id)
                                hyperopt_result_filename = FilePaths.construct_best_validation_file_name(model_prefix)

                                hyperopt_result_path = Path(file_paths['hyperopt_results_dir'], hyperopt_result_filename)
                                if hyperopt_result_path.exists():
                                    with hyperopt_result_path.open('r') as f:
                                        avg_eval_perf = float(f.readline())
                                        hyperopt_data_row = {"network_generator": network_generator,
                                                             "objective_function": objective_function,
                                                             "agent_name": agent_name,
                                                             "hyperparams_id": hyperparams_id,
                                                             "avg_perf": avg_eval_perf,
                                                             "graph_id": graph_id}

                                        hyperopt_data.append(hyperopt_data_row)

        return param_spaces, pd.DataFrame(hyperopt_data)

    def retrieve_optimal_hyperparams(self,
                                     experiment_id,
                                     model_seeds_to_skip,
                                     train_individually,
                                     return_best_loss=False):
        """Retrieve the optimal hyperparameters for a given discovery instance, objective function, and agent."""
        avg_perfs_df, param_spaces = self.get_grouped_hyp_data(experiment_id, model_seeds_to_skip, train_individually)
        gb_cols = list(set(avg_perfs_df.columns) - {"avg_perf", "hyperparams_id"})
        avg_perfs_min = avg_perfs_df.loc[avg_perfs_df.groupby(gb_cols)["avg_perf"].idxmin()].reset_index(
            drop=True)

        optimal_hyperparams = {}

        for row in avg_perfs_min.itertuples():
            if not train_individually:
                setting = row.network_generator, row.objective_function, row.agent_name
            else:
                setting = row.network_generator, row.objective_function, row.agent_name, row.graph_id
            optimal_id = row.hyperparams_id
            best_config = param_spaces[row.objective_function][row.agent_name][optimal_id]
            best_loss = row.avg_perf

            if not return_best_loss:
                optimal_hyperparams[setting] = best_config, optimal_id
            else:
                optimal_hyperparams[setting] = best_config, optimal_id, best_loss

        return optimal_hyperparams

    def get_grouped_hyp_data(self, experiment_id, model_seeds_to_skip, train_individually):
        """Retrieves hyperparameter optimization data and groups it by discovery instance, objective function, and agent."""
        param_spaces, df = self.get_hyperparameter_optimisation_data(experiment_id,
                                                                    model_seeds_to_skip,
                                                                     train_individually)
        if not train_individually:
            if 'graph_id' in df.columns:
                df = df.drop(columns='graph_id')
        avg_perfs_df = df.groupby(list(set(df.columns) - {"avg_perf"})).mean().reset_index()
        return avg_perfs_df, param_spaces

    def get_metrics_data(self, task_type, print_progress=False):
        """Retrieves metrics data for all agents within a given experiment."""
        storage_dir = self.file_paths.hyperopt_results_dir if task_type == "hyperopt" else self.file_paths.eval_results_dir
        all_results_rows = []

        all_results_files = list(storage_dir.glob("*.json"))
        for i, eval_file in enumerate(all_results_files):

            if print_progress:
                if i % 100 == 0:
                    print(f"loading file {i + 1} of {len(all_results_files)}")

            agent_name = str(eval_file.stem).split("-")[0] # convention
            agent_seed = int(str(eval_file.stem).split("-")[-2])  # convention
            with open(eval_file, "rb") as fh:
                eval_row = json.load(fh)
                eval_row["agent"] = agent_name
                eval_row["agent_seed"] = agent_seed
                all_results_rows.append(eval_row)

        return all_results_rows

    def insert_experiment_details(self,
                                    experiment_conditions,
                                    started_str,
                                    started_millis,
                                    parameter_search_spaces):
        """Serializes experiment conditions to disk so that they can be referred to later, or used in analysis notebooks."""
        all_experiment_details = {}
        all_experiment_details['experiment_id'] = self.file_paths.experiment_id
        all_experiment_details['started_datetime'] = started_str
        all_experiment_details['started_millis'] = started_millis
        all_experiment_details['file_paths'] = {k: str(v) for k, v in dict(vars(self.file_paths)).items()}

        conds = dict(vars(deepcopy(experiment_conditions)))
        del conds["agents"]

        del conds["objective_functions"]
        del conds["network_generators"]

        all_experiment_details['experiment_conditions'] = conds
        all_experiment_details['agents'] = [agent.algorithm_name for agent in experiment_conditions.agents]
        all_experiment_details['objective_functions'] = [obj.name for obj in experiment_conditions.objective_functions]
        all_experiment_details['network_generators'] = [network_generator.name for network_generator in experiment_conditions.network_generators]
        all_experiment_details['parameter_search_spaces'] = parameter_search_spaces

        import pprint
        pprint.pprint(all_experiment_details)

        with open(self.file_paths.models_dir / self.EXPERIMENT_DETAILS_FILENAME, "w") as fh:
            json.dump(all_experiment_details, fh, indent=4, sort_keys=True)

        return all_experiment_details

    def get_experiment_details(self, experiment_id):
        """Loads the experiment conditions from disk."""
        exp_models_dir = FilePaths(self.file_paths.parent_dir, experiment_id, setup_directories=False).models_dir
        with open(exp_models_dir / self.EXPERIMENT_DETAILS_FILENAME, "rb") as fh:
            exp_details_dict = json.load(fh)
        return exp_details_dict

    def store_tasks(self, tasks, task_type):
        """Stores all tasks of an experiment to disk."""
        task_storage_dir = self.file_paths.hyperopt_tasks_dir if task_type == "hyperopt" else self.file_paths.eval_tasks_dir
        count_file = self.file_paths.models_dir / f"{task_type}_tasks.count"

        count_file_out = open(count_file, 'w')
        count_file_out.write(f'{len(tasks)}\n')
        count_file_out.close()

        for task in tasks:
            out_file = task_storage_dir / FilePaths.construct_task_filename(task_type, task.task_id)
            with open(out_file, 'wb') as fh:
                pickle.dump(task, fh)

    def write_hyperopt_results(self, model_identifier_prefix, perf):
        """Writes the result of a hyperparameter optimization run to disk."""
        hyperopt_result_file = f"{self.file_paths.hyperopt_results_dir.absolute()}/" + \
                               self.file_paths.construct_best_validation_file_name(model_identifier_prefix)
        hyperopt_result_out = open(hyperopt_result_file, 'w')
        hyperopt_result_out.write('%.6f\n' % (perf))
        hyperopt_result_out.close()

    def write_metrics_dict(self, model_identifier_prefix, metrics_dict, task_type):
        """Writes the evaluated metrics to the appropriate file on disk."""
        def basic_type_converter(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()

        storage_dir = self.file_paths.hyperopt_results_dir if task_type == "hyperopt" else self.file_paths.eval_results_dir
        metrics_out_file = storage_dir / f"{model_identifier_prefix}_metrics.json"

        with open(metrics_out_file, "w") as fh:
            json.dump(metrics_dict, fh, indent=4, sort_keys=True, default=basic_type_converter)


