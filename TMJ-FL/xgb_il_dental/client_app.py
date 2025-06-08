"""xgboost_quickstart: A Flower / XGBoost app."""

from flwr.common.context import Context

import xgboost as xgb
from flwr.client import ClientApp
from flwr.common import (
    FitIns,
    FitRes,
)

from common.config import get_hyperparams_from_config
from common.XGB import load_data_dmatrix
from common.incremental import IncrementalStrategy
from xgb_dental.client_app import XgbClient


# Define Flower Client and client_fn
class XgbILClient(XgbClient):
    def __init__(
        self,
        entire_train_dmatrix : xgb.DMatrix,
        valid_dmatrix : xgb.DMatrix,
        num_local_round : int,
        xgb_params: dict,
        il_strategy: IncrementalStrategy,
    ):
        super().__init__(
            train_dmatrix=entire_train_dmatrix,
            valid_dmatrix=valid_dmatrix,
            num_local_round=num_local_round,
            xgb_params=xgb_params,   
        )
        
        # For incremental learning
        self.entire_train_dmatrix = entire_train_dmatrix
        self.il_strategy = il_strategy


    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["server_round"])
        
        self.train_dmatrix = self._get_stage_data(global_round)
        
        fit_res = super().fit(ins)
        
        il_metrics = self.il_strategy.get_il_metrics(global_round)
        for key, value in il_metrics.items():
            fit_res.metrics[key] = value

        return fit_res
    
    def _get_stage_data(self, global_round: int) -> xgb.DMatrix:

        num_points = self.entire_train_dmatrix.num_row()

        sampled_indicies = self.il_strategy.sample_data(num_points, global_round)
        # print(f"Sampled indices: {sampled_indicies}")

        return self.entire_train_dmatrix.slice(sampled_indicies)

# IL XGBoost Client
def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_local_round = context.run_config["local-epochs"]
    
    
    train_dmatrix, valid_dmatrix = load_data_dmatrix(
        partition_id, num_partitions, context.run_config
    )

    hyperparams = get_hyperparams_from_config(context.run_config, model="xgb")
    il_strategy = IncrementalStrategy(context.run_config)

    return XgbILClient(
        entire_train_dmatrix=train_dmatrix,
        valid_dmatrix=valid_dmatrix,
        num_local_round=num_local_round,
        xgb_params=hyperparams,
        il_strategy=il_strategy,
    )

# Flower ClientApp
app = ClientApp(
    client_fn,
)