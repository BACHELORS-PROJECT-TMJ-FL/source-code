"""flwr-app: A Flower / PyTorch app."""

from collections import OrderedDict

import numpy as np
import flwr as fl
import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import DataLoader 

from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, FitRes, Parameters, Metrics
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from common import report as r
from common.binary_classification import TestResultBinClassifcation
from common.save_model import save_mlp_parameters

class Net3layer(nn.Module):   

    def __init__(self, input_size=64):
        super(Net3layer, self).__init__()
        self.fc1 = nn.Linear(input_size, 40)
        self.fc2 = nn.Linear(40, 24)
        self.fc3 = nn.Linear(24, 6)
        self.fc4 = nn.Linear(6, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x) 
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
    
        return x
    
class Net2layer(nn.Module):   

    def __init__(self, input_size=64):
        super(Net2layer, self).__init__()
        self.fc1 = nn.Linear(input_size, 40)
        self.fc2 = nn.Linear(40, 6)
        self.fc3 = nn.Linear(6, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x) 
        x = torch.sigmoid(x)
    
        return x
class Net1layer(nn.Module):   

    def __init__(self, input_size=64):
        super(Net1layer, self).__init__()
        self.fc1 = nn.Linear(input_size, 40)
        self.fc2 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
    
        return x
    
class Net1largelayer(nn.Module):   

    def __init__(self, input_size=64):
        super(Net1largelayer, self).__init__()
        self.fc1 = nn.Linear(input_size, 80)
        self.fc2 = nn.Linear(80, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
    
        return x

def get_mlp(model_name: str, input_size: int) -> nn.Module:
    """Get the model based on the model name.
       Options: Net3layer, Net2layer, Net1layer, Net1largelayer
    """
    if model_name == "Net3layer":
        return Net3layer(input_size)
    elif model_name == "Net2layer":
        return Net2layer(input_size)
    elif model_name == "Net1layer":
        return Net1layer(input_size)
    elif model_name == "Net1largelayer":
        return Net1largelayer(input_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train(net: nn.Module, trainloader: DataLoader,  epochs: int, lr: float, criterion) -> float:
    """Train the model on the training set."""
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()  # Put in training mode
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            data = batch["features"]
            labels = batch["labels"]
            
            optimizer.zero_grad()   # reset from previous batch
            output = net(data)      # forward pass
            loss: torch.Tensor = criterion(output, labels) # calculate loss

            loss.backward()             # Calculate gradients
            optimizer.step()            # Update weights
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net: nn.Module, testloader: DataLoader) -> Tuple[float, str]:
    tr = TestResultBinClassifcation(criterion=torch.nn.BCELoss())

    with torch.no_grad():
        for batch in testloader:
            data = batch["features"]
            labels = batch["labels"]
            outputs = net(data)

            tr.compare_batch(outputs.data, labels.data)

    return tr.loss(), tr.to_json()


def get_weights(net: nn.Module) -> list:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)



class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        
        save_model = kwargs.pop("save_model", 0)
        self.save_model = (1 if save_model else 0)
        
        self.fold = kwargs.pop("fold", 0)
        
        self.save_at_round = kwargs.pop("save_at_round", 0)
        
        super().__init__(*args, **kwargs)
        
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, Metrics]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        r.add_fit_metrics(metrics_aggregated)

        # save parameters to file
        if self.save_model and self.save_at_round > 0 and server_round == self.save_at_round:
            if parameters_aggregated is not None:
                save_mlp_parameters(
                     self.fold, parameters_aggregated
                )
                print(f"Server round {server_round}: Model parameters saved.")
            else:
                print(f"Server round {server_round}: No parameters to save.")

        return parameters_aggregated, metrics_aggregated
        
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, Metrics]:
        loss, metrics = super().evaluate(server_round, parameters)

        r.add_central_eval_metrics(metrics)

        return loss, metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        
        loss_agg, metrics_agg = super().aggregate_evaluate(server_round, results, failures)
        
        r.add_client_eval_metrics(metrics_agg)
        
        return loss_agg, metrics_agg
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure fit for each client."""
        # Call super method to get the default configuration
        ins = super().configure_fit(server_round, parameters, client_manager)

        # Hack to get the client config
        fit_ins = ins[0][1]
        r.add_fit_config(fit_ins.config)
        
        return ins
    