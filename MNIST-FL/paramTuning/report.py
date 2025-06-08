import json
import os

from datetime import datetime

# Global file path for the report
report_path = "report.json"

def initialize_report(run_config):
    global report_path
    run_id = run_config["run-id"]

    report_path = f"report_{run_id}.json"
    
    if run_config["data.fold-cv-5"] == 1:
        fold_i = run_config["data.fold-cv-index"]
        report_path = f"report_{run_id}_fold_{fold_i}.json"
    
    # make a directory for the experiment
    experiment_name = run_config["experiment-name"]
    if experiment_name != "":
        dir = f"{experiment_name}_results"
        try:
            os.makedirs(dir, exist_ok=True)
        except FileExistsError:
            print(f"Directory {dir} already exists.")
        except Exception as e:
            print(f"An error occurred while creating the directory: {e}")
            
        report_path = f"{dir}/{report_path}"
        
    
    # print("report path init:", report_path)
    
    with open(report_path, "w", newline="") as file:
        # Print absolute path to file
        print(f"Report file path: {os.path.abspath(file.name)}")
        json.dump({}, file)

    write_to_report("run_id", run_id)
    write_to_report("experiment_name", experiment_name)
    
    _run_start_time = datetime.now()
    starttime = f"{_run_start_time.strftime('%H%M-%d%m') }"
    write_to_report("start_time", starttime)
    
    file.close()
    
def write_hyperparams(hyperparams):
    """ Write hyperparameters to the report. """
    write_to_report("hyperparams", hyperparams)

def write_to_report(key, value):
    
    data_to_append = {key: value}
    
    # print("report path from write", report_path)
    existing_data = {}
    try:
        with open(report_path, 'r') as file:
            existing_data = json.load(file)
            
        existing_data.update(data_to_append)
        
        with open(report_path, 'w') as file:
            json.dump(existing_data, file, indent=4)

    except FileNotFoundError:
        return
    except json.JSONDecodeError:
        return
    

    
    
def add_central_eval_metrics(metrics: dict):
    """ Add central metrics to the report. """
    _add_to_list("central_eval", metrics)
    
def add_fit_metrics(metrics: dict):
    """ Add fit metrics to the report. """
    _add_to_list("client_fit", metrics)
    
def add_client_eval_metrics(metrics: dict):
    """ Add client metrics to the report. """
    _add_to_list("client_eval", metrics)
    
def add_fit_config(config: dict):
    """ Add fit configuration to the report. """
    _add_to_list("fit_config", config)
    
    
def _add_to_list(key: str, value: dict):
    """ Add a dictionary to a list in the report. """
    existing_data = {}
    
    with open(report_path, 'r') as file:
        existing_data: dict = json.load(file)

        if existing_data.get(key):
            existing_data[key].append(value)
        else:
            existing_data[key] = [value]

    with open(report_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

    file.close()

def write_config(config: dict):
    """ Write the configuration to the report. """
    with open(report_path, 'r') as file:
        existing_data = json.load(file)

    existing_data["config"] = config

    with open(report_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

    file.close()