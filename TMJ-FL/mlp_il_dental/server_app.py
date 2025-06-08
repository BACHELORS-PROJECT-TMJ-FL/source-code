"""flwr-app: A Flower / PyTorch app."""

import numpy as np
import flwr as fl
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from common.MLP import CustomFedAvg
from common.incremental import get_il_config, total_il_server_rounds
from common.data import make_dataloader
from common.dental_data import get_test_data
from common.config import get_hyperparams_from_config
from common.save_model import get_saved_xgb_parameters
from mlp_dental.server_app import get_evaluate_fn, get_start_parameters, handle_eval_metrics, handle_fit_metrics

from common import report as r


def get_on_fit_config(lr: int) -> dict:
    def on_fit_config(server_round: int) -> dict:
        return {
            "lr": lr,
            "server_round": server_round,
        }
        
    return on_fit_config

# MLP il-FL
def server_fn(context: Context) -> ServerAppComponents:

    r.initialize_report(context.run_config)
    
    hyperparams = get_hyperparams_from_config(context.run_config, model="mlp")
    r.write_hyperparams(hyperparams)
    
    r.write_to_report("il_config", get_il_config(context.run_config))
    
    num_rounds = total_il_server_rounds(context.run_config)
    num_clients = context.run_config["num-clients"]
    
    
    valid_dataset = get_test_data(num_clients, context.run_config)
    valid_dataloader = make_dataloader(valid_dataset, batch_size=32)

    input_size = len(valid_dataset["features"][0])
    
    load_weights = context.run_config.get("load-weights", 0)
    fold = context.run_config.get("data.fold-cv-index", 0)
    if load_weights:
        parameters = get_saved_xgb_parameters(fold)
        r.write_to_report("load_weights", f"Loaded weights from fold {fold}")
    else: 
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
        save_model=context.run_config.get("save-model", 0),
        save_at_round=num_rounds,
        fold=fold
    )
    
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)