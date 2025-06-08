"""flwr-app: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import numpy as np
import json

from torch.utils.data import DataLoader as Dataloader
from typing import Any, Dict, Tuple
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, Metrics

from common.MLP import set_weights, get_weights, train, test, get_mlp
from common.dental_data import load_client_data
from common.data import make_dataloader



# Define Flower Client and client_fn
class MlpClient(NumPyClient):
    def __init__(self, id, net: nn.Module, train_loader: Dataloader, eval_loader: Dataloader, local_epochs: int):
        self.id = id
        self.net = net
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.local_epochs = local_epochs
        
        self.criterion = torch.nn.BCELoss()

    def fit(self, parameters, config) -> Tuple[np.ndarray, int, Dict[str, Metrics]]:
        """The last return value is a dictionary of metrics"""
        set_weights(self.net, parameters)

        train_loss = train(
            self.net,
            self.train_loader,
            self.local_epochs,
            config["lr"],
            criterion=self.criterion,
        )
        
        # send non-Metrics data like a dict as a metrics, serialize to json
        return (
            get_weights(self.net),
            len(self.train_loader.dataset),
            {"train_loss": np.round(train_loss, 3)},
        )

    # Client-side (Distributed) evaluation
    def evaluate(self, parameters, config) :
        set_weights(self.net, parameters)
        loss, metricsStr = test(self.net, self.eval_loader)
        metrics: dict = json.loads(metricsStr)
        metrics.update({"id": self.id})

        return float(loss), len(self.eval_loader.dataset), metrics



def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    train_data, test_data = load_client_data(partition_id, num_partitions, context.run_config)
    
    input_size = len(train_data["features"][0])
    net = get_mlp(context.run_config["params.mlp-name"], input_size)
    
    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]

    train_loader = make_dataloader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = make_dataloader(test_data, batch_size=batch_size, shuffle=False)
    

    return MlpClient(id=partition_id,
                     net=net,
                     train_loader=train_loader,
                     eval_loader=valid_loader,
                     local_epochs=local_epochs,
                ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
