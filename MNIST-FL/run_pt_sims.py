from paramTuning import pt

# Config for parameter tuning

pt_mlp_config = {
    "5-fold": True,
    "start_at_id": 0,
    "hyper-parameters": {
        "ILVariant": ["Task"],
        "ILStrategy": ["EWC"],
    }
}

config = {
    "run-id": 0,
    "experiment-name": "experiment3_EWC_new_task",
    "rounds-per-task": 10,
}

runs = pt.parameter_tuning(pt_mlp_config, config, run_async=True)