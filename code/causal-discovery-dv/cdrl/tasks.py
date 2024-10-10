import argparse
import json
import pickle
import time
from copy import deepcopy
import traceback
import sys
from datetime import datetime

from pathlib import Path
d = Path(__file__).resolve().parents[1]
sys.path.append(str(d.absolute()))

from cdrl.environment.graph_edge_env import DirectedGraphEdgeEnv, EnvPhase
from cdrl.evaluation.eval_utils import get_metrics_dict, extract_validation_perf_from_metrics_dict
from cdrl.io.file_paths import FilePaths
from cdrl.state.instance_generators import BNLearnInstanceGenerator
from cdrl.utils.config_utils import get_logger_instance, date_format


class OptimizeHyperparamsTask(object):
    """
    Represents a unit of execution for the hyperparameter optimization part of the experiments.
    """
    def __init__(self,
                 task_id,
                 agent,
                 objective_function,
                 network_generator,
                 experiment_conditions,
                 storage,
                 parameter_keys,
                 search_space_chunk,
                 model_seeds_chunk,
                 train_kwargs=None,
                 eval_make_action_kwargs=None,
                 additional_opts=None
                 ):
        self.task_id = task_id
        self.agent = agent
        self.objective_function = objective_function
        self.network_generator = network_generator
        self.experiment_conditions = experiment_conditions
        self.storage = storage
        self.parameter_keys = parameter_keys
        self.search_space_chunk = search_space_chunk
        self.model_seeds_chunk = model_seeds_chunk
        self.train_kwargs = train_kwargs
        self.eval_make_action_kwargs = eval_make_action_kwargs
        self.additional_opts = additional_opts

    def run(self):
        log_filename = self.storage.file_paths.construct_log_filepath()
        logger = get_logger_instance(log_filename)


        if self.network_generator == BNLearnInstanceGenerator:
            disc_inst = self.network_generator.get_instance(instance_name=self.experiment_conditions.instance_name, n=self.experiment_conditions.n, starting_graph_generation=self.experiment_conditions.starting_graph_generation)
            rfun = self.objective_function(disc_inst, **self.experiment_conditions.experiment_params)

        for model_seed in self.model_seeds_chunk:
            logger.info(f"executing for seed {model_seed}")

            exp_copy = deepcopy(self.experiment_conditions)

            for hyperparams_id, combination in self.search_space_chunk:
                hyperparams = {}

                for idx, param_value in enumerate(tuple(combination)):
                    param_key = self.parameter_keys[idx]
                    hyperparams[param_key] = param_value

                hyperparams['btm'] = exp_copy.btm_on_hyperopt
                logger.info(f"executing with hyps {hyperparams}")

                model_identifier_prefix = self.storage.file_paths.construct_model_identifier_prefix(self.agent.algorithm_name,
                                                                                                    self.objective_function.name,
                                                                                                    self.network_generator.name,
                                                                                                    model_seed, hyperparams_id)

                env = DirectedGraphEdgeEnv(disc_inst, rfun, initial_edge_budgets=exp_copy.get_initial_edge_budgets(self.network_generator, disc_inst), **exp_copy.experiment_params)
                agent_instance = self.agent(env)

                run_options = {}
                run_options["random_seed"] = model_seed
                run_options["storage"] = self.storage
                run_options["file_paths"] = self.storage.file_paths

                run_options["log_progress"] = True

                log_filename = self.storage.file_paths.construct_log_filepath()
                run_options["log_filename"] = log_filename
                run_options["model_identifier_prefix"] = model_identifier_prefix

                run_options['storage'] = self.storage
                run_options.update((self.additional_opts or {}))

                try:
                    agent_instance.setup(run_options, hyperparams)
                    if exp_copy.perform_construction:
                        construct_start = datetime.now()
                        construct_output = agent_instance.eval([disc_inst.start_state], EnvPhase.CONSTRUCT)
                        construct_end = datetime.now()
                        pruning_start_graph = construct_output[0][0]
                        duration_construct_s = (construct_end - construct_start).total_seconds()

                    else:
                        construct_output = None
                        pruning_start_graph = disc_inst.start_state
                        duration_construct_s = 0.

                    if exp_copy.perform_pruning:
                        prune_start = datetime.now()
                        prune_output = agent_instance.eval([pruning_start_graph], EnvPhase.PRUNE)
                        prune_end = datetime.now()
                        duration_prune_s = (prune_end - prune_start).total_seconds()
                    else:
                        prune_output = None
                        duration_prune_s = 0.

                    out_dict = get_metrics_dict(construct_output, prune_output, disc_inst, rfun)
                    out_dict["hyperparams"] = hyperparams
                    out_dict["hyperparams_id"] = hyperparams_id

                    out_dict["duration_construct_s"] = duration_construct_s
                    out_dict["duration_prune_s"] = duration_prune_s

                    self.storage.write_metrics_dict(model_identifier_prefix, out_dict, "hyperopt")

                    perf = extract_validation_perf_from_metrics_dict(out_dict)

                    self.storage.write_hyperopt_results(model_identifier_prefix, perf)
                    agent_instance.finalize()

                except BaseException:
                    logger.warn("got exception while training & evaluating agent")
                    logger.warn(traceback.format_exc())
                    agent_instance.finalize()


class EvaluateTask(object):
    """
    Represents a unit of execution for the evaluation part of the experiments.
    """
    def __init__(self,
                 task_id,
                 agent,
                 objective_function,
                 network_generator,
                 best_hyperparams,
                 best_hyperparams_id,
                 experiment_conditions,
                 storage,
                 model_seeds_chunk,
                 eval_make_action_kwargs=None,
                 additional_opts=None):
        self.task_id = task_id
        self.agent = agent
        self.objective_function = objective_function
        self.network_generator = network_generator

        self.best_hyperparams = best_hyperparams
        self.best_hyperparams_id = best_hyperparams_id

        self.experiment_conditions = experiment_conditions
        self.storage = storage
        self.model_seeds_chunk = model_seeds_chunk
        self.eval_make_action_kwargs = eval_make_action_kwargs
        self.additional_opts = additional_opts

    def run(self):
        log_filename = self.storage.file_paths.construct_log_filepath()
        logger = get_logger_instance(log_filename)
        local_results = []


        if self.network_generator == BNLearnInstanceGenerator:
            disc_inst = self.network_generator.get_instance(instance_name=self.experiment_conditions.instance_name, n=self.experiment_conditions.n, starting_graph_generation=self.experiment_conditions.starting_graph_generation)
            rfun = self.objective_function(disc_inst, **self.experiment_conditions.experiment_params)


        for model_seed in self.model_seeds_chunk:
            logger.info(f"executing for seed {model_seed}")
            exp_copy = deepcopy(self.experiment_conditions)
            setting = (self.network_generator.name, self.objective_function.name, self.agent.algorithm_name)

            model_identifier_prefix = self.storage.file_paths.construct_model_identifier_prefix(self.agent.algorithm_name,
                                                                                                self.objective_function.name,
                                                                                                self.network_generator.name,
                                                                                                model_seed, self.best_hyperparams_id)

            env = DirectedGraphEdgeEnv(disc_inst, rfun, initial_edge_budgets=exp_copy.get_initial_edge_budgets(self.network_generator, disc_inst), **exp_copy.experiment_params)
            agent_instance = self.agent(env)

            run_options = {}
            run_options["random_seed"] = model_seed
            run_options["file_paths"] = self.storage.file_paths
            run_options["log_progress"] = True

            log_filename = self.storage.file_paths.construct_log_filepath()
            run_options["log_filename"] = log_filename
            run_options["model_identifier_prefix"] = model_identifier_prefix

            run_options['storage'] = self.storage
            run_options.update((self.additional_opts or {}))

            try:
                hyps_copy = deepcopy(self.best_hyperparams)
                hyps_copy['btm'] = exp_copy.btm_on_eval

                agent_instance.setup(run_options, hyps_copy)
                if exp_copy.perform_construction:
                    construct_start = datetime.now()
                    construct_output = agent_instance.eval([disc_inst.start_state], EnvPhase.CONSTRUCT)
                    construct_end = datetime.now()
                    pruning_start_graph = construct_output[0][0]
                    duration_construct_s = (construct_end - construct_start).total_seconds()

                else:
                    construct_output = None
                    pruning_start_graph = disc_inst.start_state
                    duration_construct_s = 0.

                if exp_copy.perform_pruning:
                    prune_start = datetime.now()
                    prune_output = agent_instance.eval([pruning_start_graph], EnvPhase.PRUNE)
                    prune_end = datetime.now()
                    duration_prune_s = (prune_end - prune_start).total_seconds()
                else:
                    prune_output = None
                    duration_prune_s = 0.

                out_dict = get_metrics_dict(construct_output, prune_output, disc_inst, rfun)
                out_dict["hyperparams"] = hyps_copy
                out_dict["hyperparams_id"] = self.best_hyperparams_id

                out_dict["duration_construct_s"] = duration_construct_s
                out_dict["duration_prune_s"] = duration_prune_s

                self.storage.write_metrics_dict(model_identifier_prefix, out_dict, "eval")

                agent_instance.finalize()

            except BaseException:
                logger.warn("got exception while training & evaluating agent")
                logger.warn(traceback.format_exc())
                agent_instance.finalize()




def main():
    parser = argparse.ArgumentParser(description="Run a given task.")
    parser.add_argument("--experiment_part", required=True, type=str,
                        help="Whether to setup hyperparameter optimisation or evaluation.",
                        choices=["hyperopt", "eval"])

    parser.add_argument("--parent_dir", type=str, help="Root path for storing experiment data.")
    parser.add_argument("--experiment_id", required=True, help="experiment id to use")

    parser.add_argument("--task_id", type=str, required=True, help="Task id to run. Must have already been generated.")
    parser.set_defaults(parent_dir="/experiment_data")

    args = parser.parse_args()

    file_paths = FilePaths(args.parent_dir, args.experiment_id, setup_directories=False)
    task_storage_dir = file_paths.hyperopt_tasks_dir if args.experiment_part == "hyperopt" else file_paths.eval_tasks_dir
    task_file = task_storage_dir / FilePaths.construct_task_filename(args.experiment_part, args.task_id)
    with open(task_file, 'rb') as fh:
        task = pickle.load(fh)

    task.run()



if __name__ == "__main__":
    main()
































































