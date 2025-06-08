"""experiment3: A Flower / PyTorch app."""

from experiment3ERAGEM.fisher import calculate_fisher_information_domain, calculate_fisher_information_task
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from experiment3ERAGEM.task import (
    get_weights,
    set_weights,
    get_net_train_test,
    parameterRecord_to_numpy,
)
from get_data import load_data
from typing import List
import numpy as np

from flwr.common import (
    Context,
    ndarrays_to_parameters,
    array_from_numpy,
    ConfigsRecord,
)
from flwr.common import Message, RecordSet, ParametersRecord

def extract_fisher_information_matrices(message: Message) -> dict:
    """Calculate the Fisher information matrix for the given model."""
    fisher_inf_matrices = []
    for i in range(5):
        fisher_inf_matrices.append(message.content.parameters_records[f"fitins.inf_matrix_{i}"])
    # Transform ParameterRecord to numpy arrays
    fisher_inf_matrices = [parameterRecord_to_numpy(inf_matrix) if inf_matrix is not None else None for inf_matrix in fisher_inf_matrices]
    return list(filter(lambda x: len(x) > 0, fisher_inf_matrices))

def extract_optimal_client_parameters(message: Message) -> dict:
    """Calculate the Fisher information matrix for the given model."""
    opt_parameters = []
    for i in range(5):
        opt_parameters.append(message.content.parameters_records[f"fitins.client_opt_parameters_{i}"])
    # Transform ParameterRecord to numpy arrays
    opt_parameters = [parameterRecord_to_numpy(parameters) if parameters is not None else None for parameters in opt_parameters]
    return list(filter(lambda x: len(x) > 0, opt_parameters))
 
app = ClientApp()

from typing import Any, Tuple

# Initialize a global cache for dataloaders
trainloader_cache = {}
trainloader_cache_1bs = {}

# Messages with `message_type="TRAIN"` will be
# routed to this function.
@app.train()
def custom_action(message: Message, context: Context) -> Message:
    print("Executing train")
    global trainloader_cache
    global trainloader_cache_1bs

    parameters_parameterRecord: ParametersRecord = message.content.parameters_records[
        "fitins.parameters"
    ]
    gradient_parameterRecord: ParametersRecord = message.content.parameters_records[
        "fitins.gradients"
    ]

    # Extract fisher information matrices
    fisher_inf_matrices = extract_fisher_information_matrices(message)

    # Extract optimal client parameters
    optimal_client_parameters = extract_optimal_client_parameters(message)

    task_at_hand: int = message.content.configs_records["fitins.config"]["task"]
    client_id: int = context.node_config["partition-id"]

    if context.run_config["ILStrategy"] == "Fed-A-GEM":
        # If client has no data for the task, return empty message
        if task_at_hand != client_id:
            return message.create_reply(
                RecordSet(
                    configs_records={
                        "fitres.config": ConfigsRecord({"parameters_returned": False, "fisher_information_matrix": False})
                    }
                )
            )
    elif context.run_config["ILStrategy"] == "Fed-ER":
        # Terminate clients who have not yet been trained on
        if task_at_hand < client_id:
            return message.create_reply(
                RecordSet(
                    configs_records={
                        "fitres.config": ConfigsRecord({"parameters_returned": False, "fisher_information_matrix": False})
                    }
                )
            )
    elif (
        context.run_config["ILStrategy"] == "Fed-ER-0%"
        or context.run_config["ILStrategy"] == "EWC"
    ):
        if task_at_hand != client_id:
            return message.create_reply(
                RecordSet(
                    configs_records={
                        "fitres.config": ConfigsRecord({"parameters_returned": False, "fisher_information_matrix": False})
                    }
                )
            )
    elif context.run_config["ILStrategy"] == "StandardFL":
        pass  # Run train on all clients

    # Convert the parameterRecords to lists of numpy arrays
    parameters_numpy: List[np.ndarray] = parameterRecord_to_numpy(
        parameters_parameterRecord
    )
    gradient_numpy: List[np.ndarray] | None = (
        None
        if len(gradient_parameterRecord.keys()) == 0
        else parameterRecord_to_numpy(gradient_parameterRecord)
    )
    gradient_torch = (
        None
        if len(gradient_parameterRecord.keys()) == 0
        else [torch.tensor(grad) for grad in gradient_numpy]
    )

    # Get the net
    Net, _, train, _ = get_net_train_test(context.run_config["ILVariant"])

    # Instanciate the model and set its weights
    net = Net()
    set_weights(net, parameters_numpy)

    # k'th fold
    k = context.run_config["data.fold-cv-index"]

    if trainloader_cache.get(client_id) is None:
        trainloader, _ = load_data(client_id, k, dataset_name=context.run_config["dataset"])
        trainloader_cache[client_id] = trainloader
    trainloader = trainloader_cache.get(client_id)

    # Train the network for one epoch
    train_loss = train(context, net, trainloader, gradient_torch, "cpu", fisher_inf_matrices, optimal_client_parameters)

    # Extract the updated weights
    model_parameters = get_weights(net)

    res_parameters_parameterRecord = {
        str(i): array_from_numpy(NDArray) for i, NDArray in enumerate(model_parameters)
    }

    if trainloader_cache_1bs.get(client_id) is None:
        trainloader1bs, _ = load_data(client_id, k, batch_size=1, dataset_name=context.run_config["dataset"])
        trainloader_cache_1bs[client_id] = trainloader1bs
    trainloader1bs = trainloader_cache_1bs.get(client_id)

    # Calculate fisher information matrix if needed
    if context.run_config["ILStrategy"] == "EWC":
        if context.run_config["ILVariant"] == "Domain":
            fisher_information_matrix = calculate_fisher_information_domain(net, 
                trainloader1bs, torch.nn.BCELoss()
            )
        else:
            fisher_information_matrix = calculate_fisher_information_task(net, 
                trainloader1bs, torch.nn.BCELoss()
            )

        fisher_information_matrix_parameterRecord = {
            key: array_from_numpy(fisher_information_matrix[key]) for key in fisher_information_matrix.keys()
        }
    else:
        fisher_information_matrix_parameterRecord = {}

    return message.create_reply(
        RecordSet(
            parameters_records={
                "fitres.parameters": ParametersRecord(
                    array_dict=res_parameters_parameterRecord.copy()  # Copy by value instead of reference
                ),
                "fitres.information_matrix": ParametersRecord(array_dict=fisher_information_matrix_parameterRecord.copy() )
            },
            configs_records={
                "fitres.config": ConfigsRecord(
                    {
                        "parameters_returned": True,
                        "no_datapoints": len(trainloader.dataset),
                        "fisher_information_matrix": True
                    }
                )
            },
        ),
    )


def parameterRecord_to_parameters(
    parameterRecord: ParametersRecord,
) -> List[torch.Tensor]:
    """Convert a message to a gradient."""
    # Get indexes (they are strings so rather keys)
    layer_indexes = list(parameterRecord.keys())

    # Convert the bytes to a gradient
    bytes_list = [parameterRecord[i].data for i in layer_indexes]
    parameters = parameterRecord_to_numpy(bytes_list)

    return parameters


# Gradient function
@app.query()
def custom_action(message: Message, context: Context) -> Message:
    print("Executing gradient")
    global trainloader_cache

    parameters_parameterRecord: ParametersRecord = message.content.parameters_records[
        "fitins.parameters"
    ]

    # Convert the parameterRecords to lists of numpy arrays
    parameters_numpy: List[np.ndarray] = parameterRecord_to_numpy(
        parameters_parameterRecord
    )

    task_at_hand: int = message.content.configs_records["fitins.config"]["task"]
    client_id: int = context.node_config["partition-id"]

    # If client has no data for the task, return empty message
    if context.run_config["ILStrategy"] == "Fed-A-GEM":
        if client_id >= task_at_hand:
            return message.create_reply(
                RecordSet(
                    configs_records={
                        "fitres.config": ConfigsRecord({"gradient_returned": False})
                    }
                )
            )
    else:
        if True:
            return message.create_reply(
                RecordSet(
                    configs_records={
                        "fitres.config": ConfigsRecord({"gradient_returned": False})
                    }
                )
            )

    # Get the net
    Net, get_gradient, _, _ = get_net_train_test(context.run_config["ILVariant"])

    # Instanciate the model and set its weights
    net = Net()
    set_weights(net, parameters_numpy)

    # k'th fold
    k = context.run_config["data.fold-cv-index"]

    if trainloader_cache.get(client_id) is None:
        trainloader, _ = load_data(client_id, k, dataset_name=context.run_config["dataset"])
        trainloader_cache[client_id] = trainloader
    trainloader = trainloader_cache.get(client_id)

    # Train the network for one epoch
    train_loss, gradient = get_gradient(net, trainloader, "cpu")

    res_gradient_parameterRecord = {
        str(i): array_from_numpy(NDArray) for i, NDArray in enumerate(gradient)
    }

    return message.create_reply(
        RecordSet(
            parameters_records={
                "fitres.gradient": ParametersRecord(
                    array_dict=res_gradient_parameterRecord.copy()  # Copy by value instead of reference
                )
            },
            configs_records={
                "fitres.config": ConfigsRecord(
                    {
                        "gradient_returned": True,
                        "no_datapoints": len(trainloader.dataset),
                    }
                )
            },
        ),
    )
