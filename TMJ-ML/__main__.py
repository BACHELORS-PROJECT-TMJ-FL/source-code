# MLP imports
from .mlp.get_dataloaders import dataloaderSegmentations
from .get_data import k  # Get the number of folds
from .mlp.mlp_architectures import Net3layer, Net2layer, Net1layer, Net1largelayer
from .mlp.mlp_training import trainAndEvaluateNetwork
import torch
import numpy as np
import copy
from typing import Callable
import pickle
import os

# XGBoost imports
from .xgb.get_dmatrices import dmatrixSegmentations
from .xgb.xgb_training import createAndTrainModel


def kFoldCrossValidation(
    k: int,
    data_dicts: list[dict[str, any]],
    get_train_func: Callable,
    grid: dict[str, any],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    resultMatrices: list[np.ndarray] = []
    # Get results from every fold
    for fold_index in range(k):
        print(f"{fold_index=}")
        data_dict = data_dicts[fold_index]
        dataTrain10p = data_dict["train_10p"]
        dataTrain50p = data_dict["train_50p"]
        dataTrain100p = data_dict["train_100p"]
        dataTest10p = data_dict["test_10p"]
        dataTest50p = data_dict["test_50p"]
        dataTest100p = data_dict["test_100p"]
        train10p = get_train_func(dataTrain10p, dataTest10p, fold_index, device)
        train50p = get_train_func(dataTrain50p, dataTest50p, fold_index, device)
        train100p = get_train_func(dataTrain100p, dataTest100p, fold_index, device)
        resultMatrix10p: np.ndarray = gridSearch(grid, train10p, "10%")
        resultMatrix50p: np.ndarray = gridSearch(grid, train50p, "50%")
        resultMatrix100p: np.ndarray = gridSearch(grid, train100p, "100%")
        resultMatrix = np.stack([resultMatrix10p, resultMatrix50p, resultMatrix100p])
        resultMatrices.append(resultMatrix)
    # Get mean and variance of results
    resultMatrices = np.array(resultMatrices)
    mean = np.mean(resultMatrices, axis=0)
    var = np.mean((resultMatrices - mean) ** 2, axis=0)
    sd = np.sqrt(var)
    return mean, sd


def gridSearch(
    grid: dict[str, list[any]], train_func: Callable, dataSize: str
) -> Callable:
    res_matrix = gridSearchHelper(grid, train_func, dataSize)
    return res_matrix


def gridSearchHelper(
    grid: dict[str, list[any]],
    train_func: Callable,
    dataSize: str,
    parameters_for_train: dict[str, any] = {},
) -> np.ndarray | float:
    keys = list(grid.keys())
    if len(keys) == 0:
        # Add dataSize to parameters for train
        parameters_for_train["dataSize"] = dataSize
        print(f"Parameters for train: {parameters_for_train}")
        return train_func(parameters_for_train)
    reduced_grid = copy.deepcopy(grid)
    del reduced_grid[keys[0]]
    # Get deep copy of parameters for train for each val in last key of grid
    parameters_for_train_new = [
        copy.deepcopy(parameters_for_train) for val in grid[keys[0]]
    ]
    [
        params.update({keys[0]: grid[keys[0]][i]})
        for i, params in enumerate(parameters_for_train_new)
    ]
    results = np.array(
        [
            gridSearchHelper(reduced_grid, train_func, dataSize, params)
            for params in parameters_for_train_new
        ]
    )
    return results


def getMLPTrainFunction(trainloader, testloader, splitNo: int, device) -> Callable:
    def train_func(parameters: dict[str, any]) -> np.ndarray:
        netClass = parameters["nets"]
        net = netClass()
        trainloss, testloss, testaccuracy, testrecall, testprecision, testf1 = (
            trainAndEvaluateNetwork(
                net,
                parameters["epochs"],
                trainloader,
                testloader,
                parameters["learningRates"],
                device,
            )
        )
    
        return [
            testaccuracy,
            testprecision,
            testrecall,
            testf1,
            trainloss,
            testloss
        ]  # This will be the last dimension of the result matrix

    return train_func


def getXGBTrainFunction(traindmatrix, testdmatrix, splitNo: int, device) -> Callable:
    def train_func(parameters: dict[str, any]) -> np.ndarray:
        xgbparameters = {
            "objective": "binary:logistic",
            "eta": parameters["eta"],
            "max_depth": parameters["max_depth"],
            "eval_metric": ["error"],
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": parameters["subsample"],
            "tree_method": "hist",
        }
        accuracy, precision, recall, f1, auc = createAndTrainModel(
            traindmatrix, testdmatrix, xgbparameters, num_boost_round=500
        )

        # Create the directory if it doesn't exist
        return [accuracy, precision, recall, f1, auc]

    return train_func


def saveResultsToFile(
    resultMatrix: np.ndarray, grid: dict[str, any], folderName: str, meanVar: str
):
    # Add new key Value pair as the first in grid
    newGrid = {
        "dataPercentage": ["10p", "50p", "100p"],
        **{str(key): value for key, value in grid.items()},
    }

    # Create the directory if it doesn't exist
    os.makedirs(folderName, exist_ok=True)

    with open(folderName + f"final_{meanVar}_metrics.txt", "w") as f:
        saveResultsToFileHelper(resultMatrix, f, newGrid, newGrid, folderName, meanVar)


def saveResultsToFileHelper(
    result: np.ndarray | float,
    file,
    remainingGrid: dict[str, any],
    grid: dict[str, any],
    folderName: str,
    meanVar: str,
    keyValuesSelected: list[str] = [],
):
    keys = list(remainingGrid.keys())
    if len(keys) == 0:
        # Write to summary report
        startStr = ", ".join(keyValuesSelected)
        file.write(f"{startStr}: accuracy = {list(map(lambda x: x[-1], result))}\n")

        # Create the directory if it doesn't exist
        subfolder_str = "_".join(keyValuesSelected)
        os.makedirs(folderName + subfolder_str, exist_ok=True)
        with open(folderName + subfolder_str + f"/accuracy_{meanVar}.txt", "wb") as f:
            pickle.dump(result[0], f)
        with open(folderName + subfolder_str + f"/precision_{meanVar}.txt", "wb") as f:
            pickle.dump(result[1], f)
        with open(folderName + subfolder_str + f"/recall_{meanVar}.txt", "wb") as f:
            pickle.dump(result[2], f)
        with open(folderName + subfolder_str + f"/f1_{meanVar}.txt", "wb") as f:
            pickle.dump(result[3], f)
        if (folderName == "./experiment1ResultsMLP/"):
            with open(folderName + subfolder_str + f"/train_loss_{meanVar}.txt", "wb") as f:
                pickle.dump(result[4], f)
            with open(folderName + subfolder_str + f"/test_loss_{meanVar}.txt", "wb") as f:
                pickle.dump(result[5], f)
        elif (folderName == "./experiment1ResultsXGB/"):
            with open(folderName + subfolder_str + f"/auc_{meanVar}.txt", "wb") as f:
                pickle.dump(result[4], f)
        return
    keyValuesSelectedNew = [
        keyValuesSelected.copy() + [f"{keys[0]}={value if keys[0] != 'nets' else value.__name__}"]
        for value in remainingGrid[keys[0]]
    ]

    for i, keyValues in enumerate(keyValuesSelectedNew):
        remainingGridNew = copy.deepcopy(remainingGrid)
        del remainingGridNew[keys[0]]
        saveResultsToFileHelper(
            result[i], file, remainingGridNew, grid, folderName, meanVar, keyValues
        )


device = "cuda" if torch.cuda.is_available() else "cpu"

# Run MLP
MLPgrid = {
    "nets": [Net3layer, Net2layer, Net1layer, Net1largelayer],  # Net architecture
    "learningRates": [0.0001, 0.001, 0.01],  # Learning rate
    "epochs": [500],  # Epocs
}

meanResults, sdResults = kFoldCrossValidation(
    k, dataloaderSegmentations, getMLPTrainFunction, MLPgrid, device
)
saveResultsToFile(
    meanResults, MLPgrid, "./experiment1ResultsMLP/", "mean"
)
saveResultsToFile(
    sdResults, MLPgrid, "./experiment1ResultsMLP/", "var"
)

# Run XGBoost
XGBgrid = {
    "eta": [0.1, 0.01],  # Learning rate
    "max_depth": [2, 4, 6, 8, 10],  # Max depth of the trees
    "subsample": [0.5, 1],  # Subsample ratio of the training instances
}

meanResults, sdResults = kFoldCrossValidation(
    k, dmatrixSegmentations, getXGBTrainFunction, XGBgrid, device
)
saveResultsToFile(meanResults, XGBgrid, "./experiment1ResultsXGB/", "mean")
saveResultsToFile(sdResults, XGBgrid, "./experiment1ResultsXGB/", "var")
