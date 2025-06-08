"""xgboost_quickstart: A Flower / XGBoost app."""

import json
import xgboost as xgb

from typing import List, Tuple, Dict

from flwr.common import Context, Parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from common.dental_data import get_test_data
from common.data import make_dmatrix
from common.config import get_hyperparams_from_config, get_il_config
from common.XGB import CustomFedXbg, test, get_model_info
from common.incremental import total_il_server_rounds

from common import report as r
from common.save_model import get_xgb_model_as_parameters


def handle_eval_metrics(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["auc"] * num for num, metrics in eval_metrics]) / total_num
    )
    
    metrics_aggregated = {"agg.AUC": auc_aggregated}
    
    return metrics_aggregated



def on_config(server_round: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    return {
        "server_round": server_round
    }
    
def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate fit metrics from clients.
    """ 
    num_samples = [x for x, _ in metrics]
    others = [metrics for _, metrics in metrics]
    

    return {
        "num_samples": str(num_samples),
        "others": str(others),
    }

def get_evaluate_fn(valid_dmatrix, hyperparams):

    def evaluate_fn(server_round: int, parameters: Parameters, config) -> tuple[float, dict[str, float]]:
        # If at the first round, skip the evaluation
        print("Length of tensors:", len(parameters.tensors))
        if len(parameters.tensors) == 0:
            return 0, {}
        
        else: 
            bst = xgb.Booster(params=hyperparams)
            para_bytes = bytearray(parameters.tensors[0])

            # Load global model
            bst.load_model(para_bytes)
            
            m_info = get_model_info(bst)
            
            # Run evaluation
            accuracy, valid_metrics_str = test(bst, valid_dmatrix)
            valid_metrics_dict = json.loads(valid_metrics_str)
            valid_metrics_dict["round"] = server_round
            valid_metrics_dict["model_info"] = m_info
            

            return accuracy, valid_metrics_dict
    
    return evaluate_fn


# IL XGBoost Server
def server_fn(context: Context):
    
    r.initialize_report(context.run_config)
    
    hyperparams = get_hyperparams_from_config(context.run_config, "xgb")
    r.write_hyperparams(hyperparams)
    
    r.write_to_report("il_config", get_il_config(context.run_config))
    
    # Read from config
    num_rounds = total_il_server_rounds(context.run_config)
    num_clients = context.run_config["num-clients"]

    # Load
    load_weights = context.run_config.get("load-weights", 0)
    fold = context.run_config.get("data.fold-cv-index", 0)
    if load_weights:
        parameters = get_xgb_model_as_parameters(fold)
        r.write_to_report("load_weights", f"Loaded weights from fold {fold}")
    else: 
        parameters = Parameters(tensor_type="", tensors=[])
    
    
    valid_dataset = get_test_data(num_clients, context.run_config)
    valid_dmatrix = make_dmatrix(valid_dataset)


    # Define strategy
    strategy = CustomFedXbg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=num_clients,
        min_fit_clients=num_clients,
        evaluate_metrics_aggregation_fn=handle_eval_metrics,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        on_fit_config_fn=on_config,
        on_evaluate_config_fn=on_config,
        evaluate_function=get_evaluate_fn(valid_dmatrix, hyperparams),
        initial_parameters=parameters,
        save_model=context.run_config.get("save-model", 0),
        save_at_round=num_rounds,
        fold=fold,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)