def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict

def get_il_config(run_config: dict) -> dict:
    """Get incremental learning configuration from run_config."""
    il_config = {}
    for key, value in run_config.items():
        if key.startswith("il."):
            new_key = key.replace("il.", "")
            il_config[new_key] = value
            
    return il_config


def get_params_dict(run_config: dict) -> dict:
    """Get parameters dictionary from run_config."""
    params = {}
    for key, value in run_config.items():
        if key.startswith("params."):
            new_key = key.replace("params.", "")
            params[new_key] = value
            
    params = replace_keys(params)
    return params



def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

model_params = {
    "mlp": ["lr", "mlp-name",],
    "xgb": ["objective", "eta", "max_depth","max-depth", "eval-metric", "nthread", "num-parallel-tree", "subsample", "tree-method"],
}

def get_hyperparams_from_config(config: dict, model: str= "default") -> dict:
    """ Get parameters from the config dictionary. 
        Extracts variables with the prefix "params.", and based on the model type provided
    """
    
    params = {"model": model}
    
    try: 
        params_to_extract = model_params[model]
        for param in params_to_extract:
            if f"params.{param}" in config:
                params[param] = config[f"params.{param}"]
    except KeyError as e:
        print(f"Provided model '{model}' is not supported.")
        print(f"Available keys: {model_params.keys()}")    
        raise e

    if model == "xgb":
        params = replace_keys(params, match="-", target="_")
    
    return params

