import flwr as fl
import numpy as np
import xgboost as xgb

model_folder = "il_models"

def get_model_path(fold: int) -> str:
    """Get the path to the model parameters for a specific fold."""
    return f"{model_folder}/model_parameters_{fold}.npz"

def get_xgb_model_path(fold: int) -> str:
    """Get the path to the XGBoost model for a specific fold."""
    return f"{model_folder}/xgb_model_{fold}"

def get_saved_xgb_parameters(fold: int) -> fl.common.Parameters:

    path = get_model_path(fold)
    npz_file: dict = np.load(path, allow_pickle=True)
    
    ndarrays = [npz_file[key] for key in sorted(npz_file.keys())]
    parameters = fl.common.ndarrays_to_parameters(ndarrays)
    
    return parameters

def save_mlp_parameters(fold: int, parameters: fl.common.Parameters) -> None:
    path = get_model_path(fold)
    
    aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
        parameters
    )
    
    # Save aggregated_ndarrays to disk
    np.savez(path, *aggregated_ndarrays)


def save_xgb_parameters(parameters: fl.common.Parameters, fold: int) -> None:
    """Save the XGBoost model from Flower parameters to a file."""
    path = get_xgb_model_path(fold)
    
    para_bytes = parameters.tensors[0]
    
    #save bytes
    with open(path, "wb") as f:
        f.write(para_bytes)
        

def get_xgb_model_as_parameters(fold: int) -> fl.common.Parameters:
    """Load the XGBoost model from file and convert to Flower parameters."""
    path = get_xgb_model_path(fold)
    
    # Read the model bytes
    with open(path, "rb") as f:
        model_bytes = f.read()
    
    parameters = fl.common.Parameters(tensor_type="", tensors=[model_bytes])
    
    return parameters