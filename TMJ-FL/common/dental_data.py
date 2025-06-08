import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from dental_data.Pipelines import entire_data_processing_pipeline
from dental_data.FeatureEngineering import FeatureSelection as fs
from common import report as r
import pandas as pd

from common.partitioner import ClientsPartitioner
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# Cached here so that we dont load it every round
# It is both partitiond on the server and on each of the client
dental_dataset: datasets.Dataset = None  
train_partitioner = None
test_partition = None

dataprocessing_config = {
    "previous_involvement_status": "no",
    "preprocess_data": False,
    "time_slice_2_cat": ["age", 0,  18, 2000],
    "n_categories": 2,
    "feature_selection": "short",
    "encoding": "entity_embedding",
    "iterations": 51,
    "verbose": 3,
    "embedding_epochs": 20,
    "seed": 42,
}

import os
import json

def fetch_processed_data(data_path) -> datasets.Dataset:
    """Load preprocessed data from a JSON file."""
    
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found")
        return None
    
    try:
        # Load the JSON file
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        features = np.array(data["features"])
        labels = np.array(data["labels"])
        
        # Create a dataset
        ds = datasets.Dataset.from_dict({
            "features": features,
            "labels": labels,
        })
        
        print(f"Successfully loaded data from {data_path} with {len(features)} samples")
        return ds
        
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return datasets.Dataset.from_dict({
            "features": [],
            "labels": [],
        })


def preprocess_data_and_save(data_path):
    """Preprocess the data and save it to JSON files."""
    print("Preprocessing data...")
    
    # Preprocess the data
    data = entire_data_processing_pipeline(dataprocessing_config)
    
    # Get base directory
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Create a dictionary to store the processed data
    processed_data = {
        "features": data["features"],  # Convert numpy arrays to lists for JSON serialization
        "labels": data["labels"]
    }
    
    # Save the data to a JSON file
    with open(data_path, 'w') as f:
        json.dump(processed_data, f)
    
    print(f"Preprocessed data saved to {data_path}")
    
    return data  


def initialize_dataset(num_partitions: int, config: dict) -> None:
    global dental_dataset
    global test_partition
    global train_partitioner
    
    if dental_dataset is not None and train_partitioner is not None and test_partition is not None:
        return 
    
    if config["data.processed-cached"] == 1:
        # The processed data is stored in a cache or needs to be processed
        # Add logic to load the cached data if available
        
        dental_dataset = fetch_processed_data(config["data.processed-data-path"])
        print(f"Cached data loaded")
    else:   
        dental_dataset = entire_data_processing_pipeline(dataprocessing_config)
        print(f"Data processed and loaded")
    
    # Upsample-
    
    # Get features and labels
    features = dental_dataset["features"]
    labels = dental_dataset["labels"]
    
    if config["data.fold-cv-5"] == 1:
        fold_index = config["data.fold-cv-index"]
        
        if fold_index < 0 or fold_index >= 5:
            raise ValueError(f"For 5-fold CV, fold index must be 0-4, got {fold_index}")
        
        kf = KFold(n_splits=5, shuffle=False)
        
        all_splits = list(kf.split(features))
        train_indices, test_indices = all_splits[fold_index]
        
                    # Convert indices to numpy arrays of integers if they aren't already
        train_indices = np.array(train_indices, dtype=np.int64)
        test_indices = np.array(test_indices, dtype=np.int64)
        
        # Convert features and labels to numpy arrays if they aren't already
        features = np.array(features)
        labels = np.array(labels)
        
        # Now index with the integer arrays
        X_train = features[train_indices]
        y_train = labels[train_indices]
        X_test = features[test_indices]
        y_test = labels[test_indices]

        r.write_to_report("5foldCV_number", fold_index)

        test_indices_sum = np.sum(test_indices)
        print(f"SUM = {test_indices_sum}")
        r.write_to_report("test_indicies_sum", int(test_indices_sum))


    else:
        # Regular train/test split if not using cross-validation
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, shuffle=False
        )
    
    r.write_to_report("train_points:", len(X_train))
    r.write_to_report("test_points:", len(X_test))

    X_train, X_test = normalize_features(X_train, X_test)

    train_partitioner = ClientsPartitioner(num_partitions, config["data.partition-method"])
    train_partitioner.dataset = datasets.Dataset.from_dict({
        "features": X_train,
        "labels": y_train,
    })


    test_partition = datasets.Dataset.from_dict({
        "features": X_test,
        "labels": y_test,
    })

def normalize_features(X_train, X_test):
    # Convert to numpy arrays if they aren't already
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    indices_to_normalize = np.arange(19, 28)

    # Training set normalization
    # Test set should normalize based on the metrics for training set
    scaler = StandardScaler()
    X_train[:, indices_to_normalize] = scaler.fit_transform(X_train[:, indices_to_normalize])
    X_test[:, indices_to_normalize] = scaler.transform(X_test[:, indices_to_normalize])
    
    return X_train, X_test
    

def load_client_data(partition_id: int, num_partitions: int, config: dict) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Get the Dataset for a given partition_id.
    Returns a tuple of (trainset, testset)."""
    
    initialize_dataset(num_partitions, config)

    # Get the partition for the given client
    if train_partitioner is None:
        raise ValueError("Train partitioner is not initialized. Call initialize_dataset first.")
    partition = train_partitioner.load_partition(partition_id)

    # Creates training dataset
    train_set = datasets.Dataset.from_dict({
        "features": partition["features"],
        "labels": partition["labels"],
    })

    return train_set, test_partition


def get_test_data(num_partitions: int, config ):
    initialize_dataset(num_partitions, config)

    return test_partition