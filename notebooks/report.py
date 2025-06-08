import numpy as np
import json
import os

class Report:
    def __init__(self, report: dict):
        self.report = report
        
        self.clean()
    
    def clean(self):
        rounds = self.report["central_eval"]
        if not bool(rounds[0]): # skip empty rounds
            self.report["central_eval"] = rounds[1:]
    
    def extract_central_eval_metric(self, metric) -> tuple[np.ndarray, np.ndarray]:
        rounds = self.report["central_eval"]

        rounds_data = np.array([rounds[i]["round"] for i in range(len(rounds))])
        metric_data = np.array([rounds[i][metric] for i in range(len(rounds))])
        return rounds_data, metric_data


    
    def extract_fit_config_metric(self, metric) -> tuple[np.ndarray, np.ndarray]:
        rounds = self.report["fit_config"]
        print(rounds)
        rounds_data = np.array([rounds[i]["round"] for i in range(len(rounds))])
        metric_data = np.array([rounds[i][metric] for i in range(len(rounds))])
        return rounds_data, metric_data

        
class FiveFoldResults():
    def __init__(self, reports: list[Report]):
        self.reports = reports
        self.validate_reports()
        
    def validate_reports(self):
        if len(self.reports) != 5:
            raise ValueError("There should be exactly 5 reports for 5-fold cross-validation.")
        
        for i in range(1, len(self.reports)):
            if self.reports[i].report["hyperparams"] != self.reports[0].report["hyperparams"]:
                raise ValueError("Hyperparameters do not match across reports.")   
            
            if self.reports[i].report["run_id"] != self.reports[0].report["run_id"]:
                raise ValueError("Run IDs do not match across reports.")
            
    def get_client_fit(self, fold_id):
        if fold_id < 0 or fold_id >= len(self.reports):
            raise ValueError("Fold ID must be between 0 and 4.")
        
        return self.reports[fold_id].report["client_fit"]
        
            
    def get_experiment_name(self):
        return self.reports[0].report["experiment_name"]
    
    def get_start_time(self):
        return self.reports[0].report["start_time"]
        
    def hyperparameters(self):
        return self.reports[0].report["hyperparams"]
    
    def il_config(self):
        il_config = self.reports[0].report.get("il_config", None)
        ewc = il_config.get("ewc-lambda", 0) 
        replay = il_config.get("replay-percentage", 0)
        
        inc_options = f"{il_config['initial-round-percentage']*100}%-{il_config['num-initial-server-rounds']}r-{il_config['num-increments']}i-{il_config['num-rounds-per-increment']}r"
        
        return {"ewc": ewc, "replay": replay, "inc_options": inc_options}


    def get_mean_std(self, metric: str) -> tuple[np.ndarray, np.ndarray]:
        all_metrics = []
        x = None
        
        for report in self.reports:
            
            x, metric_values = report.extract_central_eval_metric(metric)
            all_metrics.append(metric_values)
            
        try:    
            # Keep a metric out of they dont have the same length as the first one
            all_metrics = [m for m in all_metrics if m.shape == all_metrics[0].shape]
            all_metrics = np.array(all_metrics)
        except ValueError as e:
            print(f"Error extracting metric '{metric}': {e}")
            
            print([r.shape for r in all_metrics])
            raise e
        
        mean = np.mean(all_metrics, axis=0)
        std = np.std(all_metrics, axis=0)
        
        return x, mean, std
    
    def get_max_final(self, metric: str):
        return self.get_max(metric), self.get_final(metric)
    
    def get_max(self, metric: str):
        _, mean, std = self.get_mean_std(metric)
        
        i = np.argmax(mean)

        return np.round(mean[i], 3), np.round(std[i], 3)

    def get_final(self, metric: str):
        _, mean, std = self.get_mean_std(metric)
        
        return np.round(mean[-1], 3), np.round(std[-1], 3)
    
def file_content(path):
    with open(path, "r") as file:
        content = file.read()
        json_content = json.loads(content)  
        file.close()
        return json_content

def select_file(folder_path, idx):
    results_files = os.listdir(folder_path)
    for i, file in enumerate(results_files):
        print(f"{i}: {file}")
    if idx < 0 or idx >= len(results_files):
        print("Invalid index. Please select a valid file index.")
        return None
    selcted_file = results_files[idx]
    print("Selected file: ", selcted_file)
    
    report = file_content(os.path.join(folder_path, selcted_file))
    
    return report

# In case of hyperparameter tuning, the report file is named differently
def report_file_name(run_id, fold_i):
    return f"report_{run_id}_fold_{fold_i}.json"


def open_single_report_file(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return None
    
    report = file_content(file_path)
    return report

def open_5fcv_report(results_folder, run_id, fold_i):
    file_name = report_file_name(run_id, fold_i)
    file_path = os.path.join(results_folder, file_name)
    return open_single_report_file(file_path)

def open_entire_5fcv_report(results_folder, run_id):
    reports = []
    for fold_i in range(5):
        file_name = report_file_name(run_id, fold_i)
        file_path = os.path.join(results_folder, file_name)
        report = open_single_report_file(file_path)
        if report is not None:
            reports.append(Report(report))
    return FiveFoldResults(reports) if reports else None

from typing import List, Optional
ExperimentReport = Optional[List[FiveFoldResults]]

def open_experiment_report(results_folder) -> ExperimentReport:
    # Open the entire 5-fold cross-validation reports for an experiment
    results_files = os.listdir(results_folder)
    runs = len(results_files) // 5  # Assuming each run has 5 folds
    if len(results_files) % 5 != 0:
        print("Warning: The number of result files is not a multiple of 5. Some folds may be missing.")

    all_reports: ExperimentReport = []
    for run in range(runs):
        reports = []
        for k in range(5):
            report = open_5fcv_report(results_folder, run, k)
            if report is None:
                continue
            
            reports.append(Report(report))
            
        FFoldResults = FiveFoldResults(reports)
        all_reports.append(FFoldResults)
    
    if len(all_reports) == 0:
        print(f"No valid reports found in folder {results_folder}")
        return None

    print(f"Found {len(all_reports)} runs in folder {results_folder.split('/')[-1]}")
    
    return all_reports


