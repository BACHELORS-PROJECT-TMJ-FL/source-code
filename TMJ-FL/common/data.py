from torch.utils.data import DataLoader
import xgboost as xgb 
import datasets
import torch

def collate_batch(batch):
    features = torch.tensor([item["features"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])

    # Ensure labels dimension is correct (should be [batch_size, 1])
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(1)

    return {
        "features": features.float(),  # Ensure float type for model
        "labels": labels.float(),  # BCE loss needs float labels
    }

def make_dataloader(dataset_dict: datasets.Dataset, **kwargs) -> DataLoader:
    """Creates a DataLoader from the dataset_dict."""
    return DataLoader(dataset_dict, **kwargs, collate_fn=collate_batch)


def make_dmatrix(data: datasets.Dataset) -> xgb.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    x = data["features"]
    y = data["labels"]
    
    # Create DMatrix with enable_categorical=True
    dmatrix = xgb.DMatrix(x, label=y, enable_categorical=True)
    return dmatrix