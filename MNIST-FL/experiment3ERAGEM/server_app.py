"""experiment3: A Flower / PyTorch app."""

from flwr.common import (
    Context,
    ndarrays_to_parameters,
    array_from_numpy,
    Metrics,
    MessageType,
    ConfigsRecord,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, Driver
from flwr.common import Message, RecordSet, ParametersRecord
import numpy as np

# from flwr.server.strategy import FedAvg
from experiment3ERAGEM.task import (
    get_weights,
    set_weights,
    get_net_train_test,
    parameterRecord_to_numpy,
)
from typing import List, Tuple, Dict
import torch
from get_data import load_data


def get_evaluate_fn(
    testloaders: Dict[str, torch.utils.data.DataLoader], context: Context, Net, test
):
    def evaluate(server_round, net, config):
        # Select the test function based on ILVariant
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)

        # Validation on all classes
        loss_global, accuracy_global, precision_global, recall_global, f1_global = test(
            net, testloaders["global"], device
        )

        # Validation on individual classes
        loss_01, accuracy_01, precision_01, recall_01, f1_01 = test(
            net, testloaders["class01"], device
        )
        loss_23, accuracy_23, precision_23, recall_23, f1_23 = test(
            net, testloaders["class23"], device
        )
        loss_45, accuracy_45, precision_45, recall_45, f1_45 = test(
            net, testloaders["class45"], device
        )
        loss_67, accuracy_67, precision_67, recall_67, f1_67 = test(
            net, testloaders["class67"], device
        )
        loss_89, accuracy_89, precision_89, recall_89, f1_89 = test(
            net, testloaders["class89"], device
        )

        metrics_dict = {
            "accuracy_global": accuracy_global,
            "accuracy_01": accuracy_01,
            "accuracy_23": accuracy_23,
            "accuracy_45": accuracy_45,
            "accuracy_67": accuracy_67,
            "accuracy_89": accuracy_89,
            "precision_global": precision_global,
            "precision_01": precision_01,
            "precision_23": precision_23,
            "precision_45": precision_45,
            "precision_67": precision_67,
            "precision_89": precision_89,
            "recall_global": recall_global,
            "recall_01": recall_01,
            "recall_23": recall_23,
            "recall_45": recall_45,
            "recall_67": recall_67,
            "recall_89": recall_89,
            "f1_global": f1_global,
            "f1_01": f1_01,
            "f1_23": f1_23,
            "f1_45": f1_45,
            "f1_67": f1_67,
            "f1_89": f1_89,
            "loss_global": loss_global,
            "loss_01": loss_01,
            "loss_23": loss_23,
            "loss_45": loss_45,
            "loss_67": loss_67,
            "loss_89": loss_89,
        }

        return metrics_dict

    return evaluate


from paramTuning.report import (
    initialize_report,
    write_hyperparams,
    add_central_eval_metrics,
)

app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    initialize_report(context.run_config)
    write_hyperparams(
        {
            "ILVariant": context.run_config["ILVariant"],
            "ILStrategy": context.run_config["ILStrategy"],
        }
    )

    fisher_inf_matrices = [None, None, None, None, None]
    saved_parameters = [None, None, None, None, None]

    Net, _, _, test = get_net_train_test(context.run_config["ILVariant"])
    # Get clients
    client_ids = driver.get_node_ids()
    # Get number of rounds
    num_rounds_per_task = context.run_config["rounds-per-task"]
    # Create instance of the model
    net = Net()
    # Init gradient
    avg_gradient = None

    # k'th fold
    k = context.run_config["data.fold-cv-index"]

    # Gets every testloader
    testloadersList = [
        testloader for _, testloader in [load_data(i, k, dataset_name=context.run_config["dataset"]) for i in range(5)]
    ]
    _, global_testloader = load_data(-1, k, dataset_name=context.run_config["dataset"])
    testloaders = {
        "global": global_testloader,
        "class01": testloadersList[0],
        "class23": testloadersList[1],
        "class45": testloadersList[2],
        "class67": testloadersList[3],
        "class89": testloadersList[4],
    }

    # Get the test/evaluation function
    eval_fn = get_evaluate_fn(testloaders, context, Net, test)

    # Perform initial central evaluation
    eval_res = eval_fn(0, net, {})
    # print(eval_res)
    add_central_eval_metrics(eval_res)

    # Start training
    for task in range(5):  # 5 tasks given by mnist
        for round in range(1, 1 + num_rounds_per_task):
            print(
                f"Round {round + task*num_rounds_per_task}/{num_rounds_per_task*5}, Task {task+1}/5"
            )
            # Get parameters of the model
            parameters = get_weights(net)

            # Send aggregated model for getting the gradients
            client_gradients, no_client_datapoints_g = prompt_client_for_gradients(
                driver, context, client_ids, parameters, task
            )

            # Save average gradient for AGEM
            avg_gradient = get_avg_gradient_parameters(
                client_gradients, no_client_datapoints_g
            )

            # Send model and average gradient for current model for training
            client_parameters, no_client_datapoints_p = prompt_client_for_parameters(
                driver,
                context,
                client_ids,
                parameters,
                avg_gradient,
                task,
                fisher_inf_matrices,
                saved_parameters,
                round
            )

            # Apply aggregation algorithm, in this case simply FedAVG
            avg_parameters = get_avg_gradient_parameters(
                client_parameters, no_client_datapoints_p
            )

            # Set the parameters of the model
            set_weights(
                net, avg_parameters
            )  # No need for aggregation, since only one client has trained on its task

            # Perform central evaluation
            eval_res = eval_fn(0, net, {})
            add_central_eval_metrics(eval_res)

            print("Round completed.")


def get_avg_gradient_parameters(
    client_gradients: List[List[np.ndarray]],
    no_client_datapoints: List[int],
) -> List[np.ndarray]:
    """Get the average gradient from the client gradients."""
    if len(client_gradients) == 0:
        return None

    total_no_datapoints = sum(no_client_datapoints)
    avg_gradient = None
    for j, gradient in enumerate(client_gradients):
        if avg_gradient is None:
            avg_gradient = gradient
            for i, g in enumerate(avg_gradient):
                avg_gradient[i] = g * no_client_datapoints[j] / total_no_datapoints
            continue
        for i, grad in enumerate(gradient):
            avg_gradient[i] += grad * no_client_datapoints[j] / total_no_datapoints

    return avg_gradient


def prompt_client_for_parameters(
    driver: Driver,
    context: Context,
    client_ids: list[int],
    net_parameters: list[torch.tensor],
    avg_gradient: list[torch.tensor],
    task: int,
    saved_fisher_inf_matrices,
    saved_parameters,
    round
) -> None:
    parametersArray = {
        str(i): array_from_numpy(NDArray) for i, NDArray in enumerate(net_parameters)
    }
    gradientsArray = (
        None
        if avg_gradient is None
        else {
            str(i): array_from_numpy(NDArray) for i, NDArray in enumerate(avg_gradient)
        }
    )

    # Create messages
    messages = []
    for client_id in client_ids:
        message: Message = driver.create_message(
            RecordSet(
                parameters_records={
                    "fitins.parameters": ParametersRecord(
                        array_dict=parametersArray.copy()  # Copy by value instead of reference
                    ),
                    "fitins.gradients": ParametersRecord(
                        array_dict=(
                            None if gradientsArray is None else gradientsArray.copy()
                        )  # Copy by value instead of reference
                    ),
                    **{
                        f"fitins.inf_matrix_{str(i)}": (
                            ParametersRecord(None) if inf_matrix is None else inf_matrix
                        )
                        for i, inf_matrix in enumerate(saved_fisher_inf_matrices)
                    },
                    **{
                        f"fitins.client_opt_parameters_{str(i)}": (
                            ParametersRecord(None) if parameters is None else parameters
                        )
                        for i, parameters in enumerate(saved_parameters)
                    },
                },
                configs_records={
                    "fitins.config": ConfigsRecord(configs_dict={"task": task})
                },
            ),
            message_type=MessageType.TRAIN,
            dst_node_id=client_id,
            group_id="test",
        )
        messages.append(message)

    # Send and wait for replies
    message_replies: list[Message] = driver.send_and_receive(messages)

    # Translate replies to a list of parameters
    filtered_replies = [
        reply for reply in message_replies if message_contains_parameters(reply)
    ]  # Should only contain one reply

    res_parameters: List[List[np.ndarray]] = [
        message_to_parameters(reply) for reply in filtered_replies
    ]

    res_no_datapoints: int = [
        message_to_no_datapoints(reply) for reply in filtered_replies
    ]

    res_fischer_information_matrix: List[dict[str, np.ndarray]] = [
        message_to_inf_matrix(reply) for reply in message_replies
    ]

    if context.run_config["ILStrategy"] == "EWC" and round % context.run_config["rounds-per-task"] == context.run_config["rounds-per-task"] - 1:
        save_fisher_information_matrix(
            saved_fisher_inf_matrices, res_fischer_information_matrix, task
        )
        save_parameters(saved_parameters, filtered_replies, task)

    return res_parameters, res_no_datapoints


def save_parameters(saved_parameters, replies, task):
    res_parameters = [
        reply.content.parameters_records["fitres.parameters"] for reply in replies
    ]
    params = res_parameters[0]
    saved_parameters[task] = params


def save_fisher_information_matrix(
    saved_fisher_inf_matrices, res_fischer_information_matrix, task
):
    fisher_inf_matrix = list(
        filter(lambda x: x is not None, res_fischer_information_matrix)
    )[0]
    saved_fisher_inf_matrices[task] = fisher_inf_matrix


def message_to_inf_matrix(reply: Message) -> dict[str, np.ndarray] | None:
    if reply.content.configs_records["fitres.config"]["fisher_information_matrix"]:
        return reply.content.parameters_records["fitres.information_matrix"]

    else:
        return None


def message_contains_parameters(reply: Message) -> bool:
    return reply.content.configs_records["fitres.config"]["parameters_returned"]


def message_to_parameters(reply: Message) -> List[torch.Tensor]:
    res_parameters_parameterRecord = reply.content.parameters_records[
        "fitres.parameters"
    ]

    res_parameters_numpy: List[np.ndarray] = parameterRecord_to_numpy(
        res_parameters_parameterRecord
    )

    return res_parameters_numpy


def message_to_no_datapoints(reply: Message) -> int:
    return reply.content.configs_records["fitres.config"]["no_datapoints"]


def prompt_client_for_gradients(
    driver: Driver,
    context: Context,
    client_ids: list[int],
    net_parameters: list[torch.tensor],
    task: int,
) -> List[List[np.ndarray]]:
    parametersArray = {
        str(i): array_from_numpy(NDArray) for i, NDArray in enumerate(net_parameters)
    }

    # Create messages
    messages = []
    for client_id in client_ids:
        message: Message = driver.create_message(
            RecordSet(
                parameters_records={
                    "fitins.parameters": ParametersRecord(
                        array_dict=parametersArray.copy()  # Copy by value instead of reference
                    )
                },
                configs_records={
                    "fitins.config": ConfigsRecord(configs_dict={"task": task})
                },
            ),
            message_type=MessageType.QUERY,
            dst_node_id=client_id,
            group_id="test",
        )
        messages.append(message)

    # Send and wait for replies
    message_replies: list[Message] = driver.send_and_receive(messages)
    filtered_replies = [
        reply for reply in message_replies if message_contains_gradient(reply)
    ]  # Should only contain one reply

    # Translate messages into gradients (each index is a gradient for the given client)
    client_gradients = [message_to_gradient(reply) for reply in filtered_replies]

    no_client_datapoints = [
        message_to_no_datapoints(reply) for reply in filtered_replies
    ]  # Should only contain one reply

    return client_gradients, no_client_datapoints


def message_contains_gradient(reply: Message) -> bool:
    return reply.content.configs_records["fitres.config"]["gradient_returned"]


def message_to_gradient(reply: Message) -> List[torch.Tensor]:
    """Convert a message to a gradient."""
    res_gradient_parameterRecord = reply.content.parameters_records["fitres.gradient"]

    res_gradient_numpy: List[np.ndarray] = parameterRecord_to_numpy(
        res_gradient_parameterRecord
    )

    return res_gradient_numpy
