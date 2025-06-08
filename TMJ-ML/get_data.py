import numpy as np
from TMJ-FL.dental_data.Pipelines import entire_data_processing_pipeline
from sklearn.preprocessing import StandardScaler
import datasets
import sys
import pandas as pd

sys.path.append("/../../../../flwrapp/")

dataprocessing_config = {
    "previous_involvement_status": "no",
    "preprocess_data": False,
    "time_slice_2_cat": ["age", 0,  18, 2000],
    "n_categories": 2,
    "feature_selection": "short",
    "encoding": "entity_embedding",
    "do_conformance_prediction": True,
    "iterations": 50,
    "verbose": 3,
    "embedding_epochs": 100,
    "seed": 42
}

# Get dataset
ds : datasets.Dataset = entire_data_processing_pipeline(dataprocessing_config, sys.path[-2] + sys.path[-1])


# Put into features and labels
X, y = ds['features'], ds['labels']

# print(X, y)
X = np.array(X, dtype=float)
y = np.array(y, dtype=float)


    
# Define k-fold cross-validation function
def kFoldCrossValidationSplit(X, y, k = 5):
    segmentSize = X.shape[0] // k
    segmentations = []
    for i in range(k):
        start = i * segmentSize
        end = (i+1) * segmentSize
        segmentation = {
            "X_train": np.concatenate([X[:start, :], X[end:, :]]),
            "y_train": np.concatenate([y[:start], y[end:]]),
            "X_test": X[start:end],
            "y_test": y[start:end],
        }
        segmentations.append(segmentation)
    return segmentations
        
# Set k for the fold crossvalidation
k = 5

# Get segmentations
segmentations = kFoldCrossValidationSplit(X, y, k)

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

# Segment the data such that some is only 10% of training and such
def createSubTrainsets(segmentations: list):
    splitSegmentations = []
    for segmentation in segmentations:
        train_len = segmentation["X_train"].shape[0]
        test_len = segmentation["X_test"].shape[0]

        X_train_100p = segmentation["X_train"]
        X_train_50p = X_train_100p[:int(0.5*train_len)]
        X_train_10p = X_train_100p[:int(0.1*train_len)]
        X_test_100p = segmentation["X_test"]
        X_test_50p = X_test_100p[:int(0.5*test_len)]
        X_test_10p = X_test_100p[:int(0.1*test_len)]

        X_train_100p, X_test_100p = normalize_features(X_train_100p, X_test_100p)
        X_train_50p, X_test_50p = normalize_features(X_train_50p, X_test_50p)
        X_train_10p, X_test_10p = normalize_features(X_train_10p, X_test_10p)

        splitSegmentation = {
            "X_train_10p": X_train_10p,
            "y_train_10p": segmentation["y_train"][:int(0.1*train_len)],
            "X_train_50p": X_train_50p,
            "y_train_50p": segmentation["y_train"][:int(0.5*train_len)],
            "X_train_100p": X_train_100p,
            "y_train_100p": segmentation["y_train"],
            "X_test_10p": X_test_10p,
            "y_test_10p": segmentation["y_test"][:int(0.1*test_len)],
            "X_test_50p": X_test_50p,
            "y_test_50p": segmentation["y_test"][:int(0.5*test_len)],
            "X_test_100p": X_test_100p,
            "y_test_100p": segmentation["y_test"],
        }
        splitSegmentations.append(splitSegmentation)
    return splitSegmentations

splitSegmentations = createSubTrainsets(segmentations)

# Done