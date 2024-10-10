import argparse
import pprint
import sys
from copy import deepcopy
from datetime import datetime

from pathlib import Path


d = Path(__file__).resolve().parents[1]
sys.path.append(str(d.absolute()))

from cdrl.evaluation.eval_utils import construct_search_spaces, generate_search_space
from cdrl.evaluation.experiment_conditions import get_conditions_for_experiment
from cdrl.io.file_paths import FilePaths
from cdrl.io.storage import EvaluationStorage
from cdrl.utils.config_utils import get_logger_instance
from cdrl.utils.general_utils import chunks
from tasks import OptimizeHyperparamsTask, EvaluateTask

"""
This script handles the creation and storage of the all the hyperparameter optimization and evaluation tasks that form an experiment.
It serializes the tasks to disk and records information about the conditions of the experiment. 
"""
def setup_hyperopt_part(experiment_conditions, args):
    """Create tasks and store data for the hyperparameter optimization part of the experiment."""
    experiment_started_datetime = datetime.now()
    started_str = experiment_started_datetime.strftime(FilePaths.DATE_FORMAT)
    started_millis = experiment_started_datetime.timestamp()

    experiment_id = args.experiment_id
    file_paths = FilePaths(args.parent_dir, experiment_id, setup_directories=True)
    storage = EvaluationStorage(file_paths)

    parameter_search_spaces = construct_search_spaces(experiment_conditions)

    storage.insert_experiment_details(
        experiment_conditions,
        started_str,
        started_millis,
        parameter_search_spaces)

    logger = get_logger_instance(file_paths.construct_log_filepath())
    setup_hyperparameter_optimisations(storage,
                                       file_paths,
                                       experiment_conditions,
                                       experiment_id,
                                       experiment_conditions.hyps_chunk_size,
                                       experiment_conditions.seeds_chunk_size)
    logger.info(
        f"{datetime.now().strftime(FilePaths.DATE_FORMAT)} Completed setting up hyperparameter optimization tasks.")


def setup_eval_part(experiment_conditions, args):
    """Create tasks and store data for the evaluation part of the experiment."""
    experiment_started_datetime = datetime.now()
    started_str = experiment_started_datetime.strftime(FilePaths.DATE_FORMAT)
    started_millis = experiment_started_datetime.timestamp()

    experiment_id = args.experiment_id
    file_paths = FilePaths(args.parent_dir, experiment_id)
    storage = EvaluationStorage(file_paths)

    parameter_search_spaces = construct_search_spaces(experiment_conditions)

    storage.insert_experiment_details(
        experiment_conditions,
        started_str,
        started_millis,
        parameter_search_spaces)

    logger = get_logger_instance(file_paths.construct_log_filepath())
    eval_tasks = construct_eval_tasks(experiment_id,
                                            file_paths,
                                            experiment_conditions,
                                            storage)

    logger.info(f"have just setup {len(eval_tasks)} evaluation tasks.")
    storage.store_tasks(eval_tasks, "eval")


def construct_eval_tasks(experiment_id,
                         file_paths,
                         original_experiment_conditions,
                         storage):
    """Create the evaluation tasks, retrieving the optimal hyperparameters and leveraging them for the agents."""
    experiment_conditions = deepcopy(original_experiment_conditions)
    logger = get_logger_instance(file_paths.construct_log_filepath())
    tasks = []
    task_id = 1

    try:
        optimal_hyperparams = storage.retrieve_optimal_hyperparams(experiment_id, {}, False)
    except (KeyError, ValueError):
        logger.warn("no hyperparameters retrieved as no configured agents require them, or no hyperparam search was carried out.")
        # logger.warn(traceback.format_exc())
        parameter_search_spaces = construct_search_spaces(experiment_conditions)
        optimal_hyperparams = {}

        for network_generator in experiment_conditions.network_generators:
            for objective_function in experiment_conditions.objective_functions:
                for agent in experiment_conditions.agents:
                    if agent.algorithm_name in parameter_search_spaces[objective_function.name]:
                        def_params = parameter_search_spaces[objective_function.name][agent.algorithm_name]['0']
                        optimal_hyperparams[(network_generator.name, objective_function.name, agent.algorithm_name)] = (def_params, 0)

    print("optimal hyperparams are:")
    pprint.pprint(optimal_hyperparams)

    for network_generator in experiment_conditions.network_generators:
        for objective_function in experiment_conditions.objective_functions:
            for agent in experiment_conditions.agents:

                additional_opts = {}
                eval_make_action_kwargs = {}

                setting = (network_generator.name, objective_function.name, agent.algorithm_name)

                if setting in optimal_hyperparams:
                    best_hyperparams, best_hyperparams_id = optimal_hyperparams[setting]
                else:
                    best_hyperparams, best_hyperparams_id = ({}, 0)

                model_seeds = experiment_conditions.test_seeds

                if agent.is_deterministic:
                    tasks.append(EvaluateTask(task_id,
                                              agent,
                                              objective_function,
                                              network_generator,
                                              best_hyperparams,
                                              best_hyperparams_id,
                                              experiment_conditions,
                                              storage,
                                              [model_seeds[0]],
                                              eval_make_action_kwargs=eval_make_action_kwargs,
                                              additional_opts=additional_opts
                                              ))
                    task_id += 1


                else:
                    model_seeds_chunks = list(chunks(model_seeds, experiment_conditions.seeds_chunk_size))

                    for model_seeds_chunk in model_seeds_chunks:
                        tasks.append(EvaluateTask(task_id,
                                                agent,
                                                objective_function,
                                                network_generator,
                                                best_hyperparams,
                                                best_hyperparams_id,
                                                experiment_conditions,
                                                storage,
                                                model_seeds_chunk,
                                                eval_make_action_kwargs=eval_make_action_kwargs,
                                                additional_opts=additional_opts
                                                ))
                        task_id += 1

    return tasks


def setup_hyperparameter_optimisations(storage,
                                       file_paths,
                                       experiment_conditions,
                                       experiment_id,
                                       hyps_chunk_size,
                                       seeds_chunk_size):
    """Create the hyperparameter optimization tasks that explore the parameter space for each agent."""
    model_seeds = experiment_conditions.validation_seeds

    hyperopt_tasks = []

    start_task_id = 1
    for network_generator in experiment_conditions.network_generators:
        for obj_fun in experiment_conditions.objective_functions:
            for agent in experiment_conditions.agents:
                if agent.algorithm_name in experiment_conditions.hyperparam_grids[obj_fun.name]:
                    agent_param_grid = experiment_conditions.hyperparam_grids[obj_fun.name][agent.algorithm_name]

                    local_tasks = construct_parameter_search_tasks(
                            start_task_id,
                            agent,
                            obj_fun,
                            network_generator,
                            experiment_conditions,
                            storage,
                            file_paths,
                            agent_param_grid,
                            model_seeds,
                            experiment_id,
                            hyps_chunk_size,
                            seeds_chunk_size)
                    hyperopt_tasks.extend(local_tasks)
                    start_task_id += len(local_tasks)

    logger = get_logger_instance(file_paths.construct_log_filepath())
    logger.info(f"created {len(hyperopt_tasks)} hyperparameter optimisation tasks.")
    storage.store_tasks(hyperopt_tasks, "hyperopt")


def construct_parameter_search_tasks(start_task_id,
                                     agent,
                                     objective_function,
                                     network_generator,
                                     experiment_conditions,
                                     storage,
                                     file_paths,
                                     parameter_grid,
                                     model_seeds,
                                     experiment_id,
                                     hyps_chunk_size,
                                     seeds_chunk_size):
    """Groups hyperparameter and seeds choices into chunks to be executed together in a single task."""
    parameter_keys = list(parameter_grid.keys())
    local_tasks = []
    search_space = list(generate_search_space(parameter_grid).items())

    search_space_chunks = list(chunks(search_space, hyps_chunk_size))
    model_seed_chunks = list(chunks(model_seeds, seeds_chunk_size))

    print(search_space_chunks)
    print(model_seed_chunks)

    eval_make_action_kwargs = {}
    additional_opts = {}

    task_id = start_task_id
    for ss_chunk in search_space_chunks:
        for ms_chunk in model_seed_chunks:
            local_tasks.append(OptimizeHyperparamsTask(task_id,
                                                           agent,
                                                           objective_function,
                                                           network_generator,
                                                           experiment_conditions,
                                                           storage,
                                                           parameter_keys,
                                                           ss_chunk,
                                                           ms_chunk,
                                                           additional_opts=additional_opts,
                                                           eval_make_action_kwargs=eval_make_action_kwargs))
            task_id += 1

    return local_tasks

def main():
    parser = argparse.ArgumentParser(description="Setup tasks for experiments.")
    parser.add_argument("--experiment_part", required=True, type=str,
                        help="Whether to setup hyperparameter optimisation or evaluation.",
                        choices=["hyperopt", "eval"])

    parser.add_argument("--which", required=True, type=str,
                        help="Which experiment to run",
                        choices=["main"])

    parser.add_argument("--parent_dir", type=str, help="Root path for storing experiment data.")
    parser.add_argument("--experiment_id", required=True, help="experiment id to use")

    parser.add_argument("--instance_name", type=str, required=True, help="Causal discovery problem instance.")

    parser.add_argument("--budget", type=float, required=False, help="Simulation budget modifier.")

    parser.set_defaults(parent_dir="/experiment_data")

    args = parser.parse_args()

    experiment_conditions = get_conditions_for_experiment(args.which, args.instance_name, args)

    if args.experiment_part == "hyperopt":
        setup_hyperopt_part(experiment_conditions, args)
    elif args.experiment_part == "eval":
        setup_eval_part(experiment_conditions, args)


if __name__ == "__main__":
    main()
