"""experiment3: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flwr.common import ndarray_to_bytes, bytes_to_ndarray, Array
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def gradient_to_bytes_list(gradient: list[torch.Tensor]):
    """Convert a gradient to a list of bytes."""
    gradient_bytes = [ndarray_to_bytes(grad.cpu().numpy()) for grad in gradient]
    return gradient_bytes


def parameters_to_bytes_list(parameters: list[torch.Tensor]):
    """Convert a gradient to a list of bytes."""
    parameters_bytes = [ndarray_to_bytes(params.cpu().numpy()) for params in parameters]
    return parameters_bytes


def parameterRecord_to_numpy(parameterRecord: OrderedDict[Array]):
    """Convert a list of bytes to a gradient."""
    return [value.numpy() for value in parameterRecord.values()]


# Loss function used for EWC
class EWCLoss(nn.Module):
    def __init__(
        self,
        base_criterion,
        importance_weights,
        optimal_client_parameters,
        model,
        lamb=1e2
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.importance_weights = importance_weights
        self.optimal_client_parameters = optimal_client_parameters
        self.lamb = lamb
        self.current_model: nn.Module = model

    def forward(self, output, target):
        # Standard task loss
        loss = self.base_criterion(output, target)

        if len(self.optimal_client_parameters) == 0:
            return loss

        # Add EWC penalty
        ewc_loss = 0
        # Get weights of self.current_model
        model_parameters = [
            param for param in self.current_model.parameters() if param.requires_grad
        ]

        for i in range(len(self.importance_weights)):
            importance_weight = self.importance_weights[i]
            opt_params = self.optimal_client_parameters[i]
            for j in range(len(importance_weight)):
                ws = torch.tensor(importance_weight[j])
                ops = torch.tensor(opt_params[j])
                ewc_loss += (ws * (model_parameters[j] - ops).pow(2)).sum()

        # print("EWC loss:", ewc_loss.item(), "Loss:", loss.item())

        return loss + self.lamb * 1/2 * ewc_loss


def getLossFunction(
    context,
    base_criterion: nn.Module,
    model,
    importance_weights: List = None,
    optimal_client_parameters: List = None,
):
    """Get the loss function based on the IL strategy."""
    if context.run_config["ILStrategy"] == "EWC":
        return EWCLoss(
            base_criterion, importance_weights, optimal_client_parameters, model, lamb=1e2 if context.run_config["ILVariant"] == "Task" else 1e4
        )
    else:
        return base_criterion


class NetDomainIL(nn.Module):
    def __init__(self):
        super(NetDomainIL, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def get_gradient_DomainIL(net: nn.Module, trainloader, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    net.train()

    cumulated_gradient = None

    for batch in trainloader:
        # Get images and labels
        images = batch["image"].to(device)
        labels = (batch["label"] % 2).to(device)  # Convert to binary labels (0 or 1)
        labels = labels.cpu().type(torch.float32)[:, torch.newaxis]

        # Reset the gradient
        optimizer.zero_grad()

        # Get the loss and compute the gradient
        loss = criterion(net(images.to(device)), labels.to(device))
        loss.backward()

        # extract gradient
        gradient: list[torch.tensor] = [
            param.grad for param in net.parameters() if param.grad is not None
        ]
        cumulated_gradient = (
            gradient
            if cumulated_gradient is None
            else [cumulated_gradient[i] + gradient[i] for i in range(len(gradient))]
        )

    avg_gradient = [grad / len(trainloader) for grad in cumulated_gradient]

    return loss.item(), [gradient.cpu().numpy() for gradient in avg_gradient]


def input_gradient_to_net(net: nn.Module, gradient, device):
    for i, param in enumerate(net.parameters()):
        param.grad = torch.tensor(gradient[i]).to(device)


def flatten_gradient(gradient: list[torch.tensor]) -> np.array:
    return np.concatenate([g.numpy().flatten() for g in gradient], axis=0)


def restructure_gradient(
    gradient: np.array, shape: list[tuple[int]]
) -> list[torch.tensor]:
    restructured_gradient = []
    for ind_shape in shape:
        # Get gradients of first layer
        temp_grad = gradient[: np.prod(ind_shape)]
        # Remove used gradients
        gradient = gradient[np.prod(ind_shape) :]
        # Reshape first gradient
        temp_grad = temp_grad.reshape(ind_shape)
        restructured_gradient.append(torch.tensor(temp_grad))
    return restructured_gradient


def orthogonalizeGradient(
    gradient: list[torch.tensor], GEM_gradient: list[torch.tensor]
):
    gradient_flat = flatten_gradient(gradient)
    GEM_gradient_flat = flatten_gradient(GEM_gradient)

    res_gradient = gradient_flat.copy()

    cosAlpha = np.dot(gradient_flat, GEM_gradient_flat) / (
        np.linalg.norm(gradient_flat) * np.linalg.norm(GEM_gradient_flat)
    )

    gradient_shape = [g.shape for g in gradient]

    return (
        gradient
        if cosAlpha > 0
        else restructure_gradient(
            res_gradient
            - (
                np.dot(res_gradient, GEM_gradient_flat)
                / np.linalg.norm(GEM_gradient_flat) ** 2
            )
            * GEM_gradient_flat,
            gradient_shape,
        )
    )


def trainDomainIL(
    context,
    net: nn.Module,
    trainloader,
    GEM_gradient: List[torch.tensor],
    device,
    fisher_information_matrices=None,
    optimal_client_parameters=None,
):
    net.to(device)  # move model to GPU if available
    net.train()
    # Create optimizer
    criterion = getLossFunction(
        context,
        torch.nn.BCELoss().to(device),
        net,
        importance_weights=fisher_information_matrices,
        optimal_client_parameters=optimal_client_parameters,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    optimizer.zero_grad()
    cumulated_loss = 0.0

    # Update gradient in the model
    for batch in trainloader:
        images = batch["image"].to(device)
        labels = (batch["label"] % 2).to(device)
        labels = labels.cpu().type(torch.float32)[:, torch.newaxis]

        optimizer.zero_grad()

        # Get the loss and compute the gradient
        loss = criterion(net(images), labels)
        loss.backward()

        # AGEM if gradient is not None
        if GEM_gradient is not None:
            # Extract gradient from the net
            gradient = [
                param.grad for param in net.parameters() if param.grad is not None
            ]

            # Get the gradient after orthogonalization
            orthogonalizedGradient = orthogonalizeGradient(gradient, GEM_gradient)

            # Set the gradient to the model
            input_gradient_to_net(net, orthogonalizedGradient, device)

        # Apply gradient
        optimizer.step()

        cumulated_loss += loss.item()

    return cumulated_loss / len(trainloader)


def testDomainIL(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.BCELoss()
    loss = 0.0
    all_outputs = np.array([])[:, np.newaxis]
    all_labels = np.array([])[:, np.newaxis]
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = (batch["label"] % 2).to(
                device
            )  # Convert to binary labels (0 or 1)
            labels = labels.cpu().type(torch.float32)[:, torch.newaxis]
            outputs = net(images)
            loss += criterion(outputs, labels).item()

            all_outputs = np.vstack([all_outputs, outputs.cpu().numpy()])
            all_labels = np.vstack([all_labels, labels.cpu().numpy()])
    accuracy = accuracy_score(all_labels, all_outputs > 0.5)
    precision = precision_score(all_labels, all_outputs > 0.5, average="binary")
    recall = recall_score(all_labels, all_outputs > 0.5, average="binary")
    f1 = f1_score(all_labels, all_outputs > 0.5, average="binary")
    loss = loss / len(testloader)
    return loss, accuracy, precision, recall, f1


class NetTaskIL(nn.Module):
    def __init__(self):
        super(NetTaskIL, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def get_gradient_TaskIL(net: nn.Module, trainloader, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    net.train()

    cumulated_gradient = None

    for batch in trainloader:
        labels = batch["label"]
        # convert labels to 5d vector with 0, 1 being the first index
        labels_handled = np.zeros((len(labels), 5), dtype=np.float32)
        for i in range(len(labels)):
            index = labels[i] // 2
            labels_handled[i][index] = labels[i] % 2
        labels = torch.tensor(labels_handled)
        images = batch["image"]
        outputs = net(images)
        labels = labels.cpu().type(torch.float32)

        # Reset the gradient
        optimizer.zero_grad()

        # Get the loss and compute the gradient
        loss = criterion(outputs, labels.to(device))
        loss.backward()

        # extract gradient
        gradient: list[torch.tensor] = [
            param.grad for param in net.parameters() if param.grad is not None
        ]
        cumulated_gradient = (
            gradient
            if cumulated_gradient is None
            else [cumulated_gradient[i] + gradient[i] for i in range(len(gradient))]
        )

    avg_gradient = [grad / len(trainloader) for grad in cumulated_gradient]

    return loss.item(), [gradient.cpu().numpy() for gradient in avg_gradient]


def trainTaskIL(
    context,
    net: nn.Module,
    trainloader,
    GEM_gradient: List[torch.tensor],
    device,
    fisher_information_matrices=None,
    optimal_client_parameters=None,
):
    net.to(device)  # move model to GPU if available
    net.train()

    # Create optimizer
    criterion = getLossFunction(
        context,
        torch.nn.BCELoss().to(device),
        net,
        importance_weights=fisher_information_matrices,
        optimal_client_parameters=optimal_client_parameters,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    optimizer.zero_grad()
    cumulated_loss = 0.0

    # Update gradient in the model
    for batch in trainloader:
        images = batch["image"].to(device)
        labels = batch["label"]
        # convert labels to 5d vector with 0, 1 being the first index
        labels_handled = np.zeros((len(labels), 5), dtype=np.float32)
        indexes = []
        for i in range(len(labels)):
            index = labels[i] // 2
            labels_handled[i][index] = labels[i] % 2
            indexes.append(index)
        labels = torch.tensor(labels_handled)
        outputs = net(images)
        # Convert non needed outputs of 5d vector to zero
        outputs_extracted = outputs[range(outputs.shape[0]), indexes]
        outputs = torch.tensor(np.zeros((outputs.shape[0], 5), dtype=np.float32))
        outputs[range(outputs.shape[0]), indexes] = outputs_extracted
        labels = labels.cpu().type(torch.float32)

        optimizer.zero_grad()

        # Get the loss and compute the gradient
        loss = criterion(outputs, labels)
        loss.backward()

        # AGEM if gradient is not None
        if GEM_gradient is not None:
            # Extract gradient from the net
            gradient = [
                param.grad for param in net.parameters() if param.grad is not None
            ]

            # Get the gradient after orthogonalization
            orthogonalizedGradient = orthogonalizeGradient(gradient, GEM_gradient)

            # Set the gradient to the model
            input_gradient_to_net(net, orthogonalizedGradient, device)

        # Apply gradient
        optimizer.step()

        cumulated_loss += loss.item()

    return cumulated_loss / len(trainloader)


def testTaskIL(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.BCELoss()
    loss = 0.0
    all_outputs = np.empty((0, 5))
    all_labels = np.empty((0, 5))
    indexes = []
    with torch.no_grad():
        for batch in testloader:
            labels = batch["label"]
            # convert labels to 5d vector with 0, 1 being the first index
            labels_handled = np.zeros((len(labels), 5), dtype=np.float32)
            for i in range(len(labels)):
                index = labels[i] // 2
                labels_handled[i][index] = labels[i] % 2
                indexes.append(index)
            labels = torch.tensor(labels_handled)
            images = batch["image"].to(device)
            outputs = net(images)
            labels = labels.cpu().type(torch.float32)
            loss += criterion(outputs, labels).item()

            all_outputs = np.vstack([all_outputs, outputs.cpu().numpy()])
            all_labels = np.vstack([all_labels, labels.cpu().numpy()])
    # Get the indexes specified by indexes in all rows in outputs and all_labels
    all_outputs = (
        all_outputs[range(all_outputs.shape[0]), indexes] > 0.5
    )  # Ensure outputs are 2D
    all_labels = all_labels[range(all_labels.shape[0]), indexes]  # Ensure labels are 2D
    accuracy = accuracy_score(all_labels, all_outputs)
    precision = precision_score(all_labels, all_outputs, average="binary")
    recall = recall_score(all_labels, all_outputs, average="binary")
    f1 = f1_score(all_labels, all_outputs, average="binary")
    loss = loss / len(testloader)
    return loss, accuracy, precision, recall, f1


class NetClassIL(nn.Module):
    def __init__(self):
        super(NetClassIL, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(
            x
        )  # cross entropy loss expects raw logits not probabilities, thus no softmax here


def trainClassIL(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def testClassIL(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    x = net.load_state_dict(state_dict, strict=True)


def get_net_train_test(ILVariant: str):
    """Get the model and train/test functions based on the IL variant."""
    if ILVariant == "Domain":
        net = NetDomainIL
        get_gradient = get_gradient_DomainIL
        train = trainDomainIL
        test = testDomainIL
    elif ILVariant == "Task":
        net = NetTaskIL
        get_gradient = get_gradient_TaskIL
        train = trainTaskIL
        test = testTaskIL
    elif ILVariant == "Class":
        net = NetClassIL
        get_gradient = None  # TODO: Change to get_gradient_ClassIL
        train = trainClassIL
        test = testClassIL
    else:
        raise ValueError(f"Selected ILVariant is not supported: {ILVariant}.")

    return net, get_gradient, train, test
