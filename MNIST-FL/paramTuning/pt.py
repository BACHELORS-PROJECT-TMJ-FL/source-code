from typing import Dict, List
from .flwr_stuff import run_flower_simulation, run_flower_simulation_async
import datetime

from itertools import product
from .typing import Config, HPTConfig

class PTResult:
    def __init__(self, run_id: int, fold_successes: List[bool], hyperparams: Dict[str, any], runtime: float = 0.0):
        

        self.run_id = run_id
        self.fold_successes = fold_successes
        self.hyperparams = hyperparams
        self.runtime = runtime

def print_pt_results(pt_runs: List[PTResult]):
    """ Print the results of the parameter tuning. """
    print(" ==== Parameter Tuning Results: ====")
    for result in pt_runs:
        print(f"Run ID: {result.run_id}, Success: {result.fold_successes}, Hyperparameters: {result.hyperparams}, Runtime: {result.runtime}")


def parameter_tuning(pt_config: Config,
                     config: Config,
                     print_results = True,
                     run_async = False,
                     ) -> List[PTResult]:
    
    pt_runs: List[PTResult] = []
    hyperparams = pt_config.get("hyper-parameters", {})
    combinations = get_cartesian_product(hyperparams)

    start_at_id = pt_config.get("start_run", 0)

    print(f"Total combinations: {len(combinations)}")

    # Check if 5-fold cross-validation is enabled
    folds = 1
    if pt_config.get("5-fold", False):
        folds = 5
        config["data.fold-cv-5"] = 1
    else:
        config["data.fold-cv-5"] = 0
      
    for run_id, hp_combination in enumerate(combinations):
        # Skip the first 'start_at_id' runs
        print("KIG HER", run_id, start_at_id)
        if run_id < start_at_id:
            continue

        config["run-id"] = run_id
        successes = [False] * folds
        start_time = datetime.datetime.now()
        
        print(f"\nRun ID: {run_id}")
        print(f"Running hyperparameter combination: {hp_combination}")
        print("Starting at:", start_time)

        for hp_key, value in hp_combination.items():
            # Update the config with the new parameter value
            config[hp_key] = value
        
        if run_async:
            processes = []
            for i in range(folds):
                config["data.fold-cv-index"] = i
                process = run_flower_simulation_async(config)
                
                print(f"Started process for fold {i} with PID: {process.pid}")

                processes.append(process)

            # Wait for all processes to finish
            for i, process in enumerate(processes):
                return_code = process.wait()
                successes[i] = (return_code == 0)
                print(f"Process {i} completed with return code: {return_code}")
        else:
            for i in range(folds):
                config["data.fold-cv-index"] = i
                success = run_flower_simulation(config)
                successes[i] = success
                print(f"Fold {i} completed with success: {success}")

        # Reset the parameter to its original value for the next iteration
        curr_hyperparams = get_curr_pt_config(hyperparams, config)
        elapsed_time = datetime.datetime.now() - start_time

        result = PTResult(
            run_id=run_id,
            fold_successes=successes,
            hyperparams=curr_hyperparams,
            runtime=elapsed_time
        )
            
        pt_runs.append(result)
        
        print(f"Run ID: {result.run_id}, Success: {result.fold_successes}, Hyperparameters: {result.hyperparams}, Runtime: {result.runtime}")
            
    if print_results:
        print_pt_results(pt_runs)
    
    return pt_runs




def get_cartesian_product(hyperparams: HPTConfig) -> List[Config]:
    """ Generate all combinations of hyperparameters. """
    
    keys = hyperparams.keys()
    values = (hyperparams[key] for key in keys)
    
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    return combinations

def get_curr_pt_config(hyperparams, config: dict):
    current_pt_config = {key: value for key, value in config.items() if key in hyperparams}
    return current_pt_config

