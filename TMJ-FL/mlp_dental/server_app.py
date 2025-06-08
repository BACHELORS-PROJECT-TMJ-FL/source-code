"""flwr-app: A Flower / PyTorch app."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
import torch


from common.MLP import ( CustomFedAvg,  
                       get_mlp,
                test,
                set_weights,
                get_weights)
from common.data import make_dataloader
from common.dental_data import get_test_data
from common.config import get_hyperparams_from_config

from common import report as r

import json

# Centralized evaluation,
def get_evaluate_fn(valid_loader, net_name, input_size) -> callable:
    """Return a function that evaluates the global model on the test set."""

    def evaluate(server_round, parameters, config):
        net = get_mlp(net_name, input_size)
        set_weights(net, parameters)
        
        valid_loss, valid_metrics_str = test(net, valid_loader)
        
        valid_metrics_dict = json.loads(valid_metrics_str)
        valid_metrics_dict["round"] = server_round
     
        return valid_loss, valid_metrics_dict

    return evaluate


def get_on_fit_config(lr: int) -> dict:
    def on_fit_config(server_round: int) -> dict:
        return {
            "lr": lr,
            "server_round": server_round,
        }
        
    return on_fit_config



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

def handle_eval_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate client metrics. e.g. their averages

    Metrics are given as dictionary from the client-side evaluate function
    """
    
    per_client_metrics = [
                            [metrics["id"], test_points, metrics["accuracy"]]
                            for test_points, metrics in metrics
                         ]

    return {
        "metrics": per_client_metrics,
    }

def get_start_parameters(model_name: str, input_size: int) -> torch.nn.Parameter:
    ndarrays = get_weights(get_mlp(model_name, input_size))
    parameters = ndarrays_to_parameters(ndarrays)

    return parameters

# MLP Standard-FL
def server_fn(context: Context) -> ServerAppComponents:

    r.initialize_report(context.run_config)
    
    hyperparams = get_hyperparams_from_config(context.run_config, model="mlp")
    r.write_hyperparams(hyperparams)
    
    num_rounds = context.run_config["num-server-rounds"]
    num_clients = context.run_config["num-clients"]
    
    valid_dataset = get_test_data(num_clients, context.run_config)
    valid_dataloader = make_dataloader(valid_dataset, batch_size=32)

    input_size = len(valid_dataset["features"][0])
    parameters = get_start_parameters(hyperparams["mlp-name"], input_size)
    
    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=num_clients,
        min_fit_clients=num_clients,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=handle_eval_metrics,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        on_fit_config_fn=get_on_fit_config(hyperparams["lr"]),
        evaluate_fn=get_evaluate_fn(valid_dataloader, net_name=hyperparams["mlp-name"], input_size=input_size),
    )
    
    config = ServerConfig(num_rounds=num_rounds, round_timeout=1200)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)


