from fl_parameter_tuning import pt
from common.dental_data import preprocess_data_and_save 

# Config for parameter tuning

pt_mlp_config = {
    "5-fold": True,
    "hyper-parameters": {
        "params.lr": [0.001],
        "params.mlp-name": ["Net1layer"],
        # "il.ewc-lambda": [0, 1e3, 1e4, 1e5, 1e6],
    }
}

# hyperparams for MLP
# 'params.lr': [0.001,0.0001],
# "params.mlp-name": ["Net3layer", "Net2layer", "Net1layer", "Net1largelayer"]

pt_xgb_config = {
    "5-fold": True,
    "hyper-parameters": {
        "params.eta": [0.01],
        "params.max-depth": [6],
        "params.subsample": [1.0],
        # "il.replay-percentage": [0.0, 0.1, 0.3, 1.0],
    }
}

# hyperparameters xgb
# "params.eta": [0.01,  0.1],
# "params.max-depth": [2, 4, 6],
# "params.subsample": [0.5, 1.0],

preprocess_path = "./dental_data/Data/processed_data.json"

# Config for the simulation
config = {
    "run-id": 0,
    "experiment-name": "exp2-xgb-6c-v2-initial2",
    "num-server-rounds": 100,
    "local-epochs": 2,
    "num-clients": 6,
    "data.processed-cached": 1,
    "data.processed-data-path": preprocess_path,
}

preprocess_data_and_save(preprocess_path)

runs = pt.parameter_tuning(pt_xgb_config, config, run_async=False)
