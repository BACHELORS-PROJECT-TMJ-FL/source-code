
import torch.nn as nn
import numpy as np

from datasets import Dataset
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from flwr.client import ClientApp
from flwr.common import Context, Metrics

from common.MLP import get_mlp, set_weights
from common.dental_data import load_client_data
from common.data import make_dataloader
from common.incremental import IncrementalStrategy

from mlp_dental.client_app import MlpClient


# Define Flower Client and client_fn
class MlpILClient(MlpClient):
    def __init__(self, id, net: nn.Module, train_dataset: Dataset, eval_loader: DataLoader, local_epochs: int, il_strategy: IncrementalStrategy):
        
        self.train_dataset = train_dataset
        self.il_strategy = il_strategy
        train_loader = make_dataloader(train_dataset, batch_size=32, shuffle=True)
        super().__init__(id, net, train_loader, eval_loader, local_epochs)
        
        self.previous_stage = 0


    def fit(self, parameters, config) -> Tuple[np.ndarray, int, Dict[str, Metrics]]:
        """The last return value is a dictionary of metrics"""
        global_round = int(config["server_round"])
        
        stage = self.il_strategy.get_il_stage(global_round)
        
        set_weights(self.net, parameters)
        
        # If moving to a new stage, calculate Fisher on the current data
        # before sampling new data for the next stage
 
        
        num_points = self.train_dataset.num_rows
        if self.previous_stage != stage and stage > 0:
            
            fisher_loader = make_dataloader(self.train_dataset.select(
                                                                      self.il_strategy.data_until_stage(num_points, stage)
                                                                      ), batch_size=1, shuffle=True)
            self.criterion = self.il_strategy.get_criterion(self.net, fisher_loader, stage)
            
        metrics = {}
        # Sample data
        sampled_indices = self.il_strategy.sample_data(num_points, global_round)
        
        self.train_loader = make_dataloader(self.train_dataset.select(sampled_indices), batch_size=32, shuffle=True)
        
        self.previous_stage = stage
        
        # Call the original fit method
        weights, n, fitmetrics = super().fit(parameters, config)
        
        metrics.update(fitmetrics)

        data_indices_sum = sampled_indices.sum()
        print(f"Sampled indices: {data_indices_sum}")

        il_metrics = self.il_strategy.get_il_metrics(global_round)
        for key, value in il_metrics.items():
            metrics[key] = value
        
        metrics["sampled_indices_sum"] = int(data_indices_sum)
            

        return weights, n, metrics


def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    
    train_data, test_data = load_client_data(partition_id, num_partitions, context.run_config)
    
    input_size = len(train_data["features"][0])
    net = get_mlp(context.run_config["params.mlp-name"], input_size)
    
    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]

    valid_loader = make_dataloader(test_data, batch_size=batch_size, shuffle=False)
    
    il_strategy = IncrementalStrategy(context.run_config)

    return MlpILClient(id=partition_id,
                     net=net,
                     train_dataset=train_data,
                     eval_loader=valid_loader,
                     local_epochs=local_epochs,
                     il_strategy=il_strategy
                ).to_client()



# Flower ClientApp
app = ClientApp(
    client_fn,
)
