"""xgboost_quickstart: A Flower / XGBoost app."""

import xgboost as xgb

from flwr.common.context import Context
from flwr.client import Client, ClientApp
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
)

from common.config import replace_keys, get_hyperparams_from_config
from common.XGB import load_data_dmatrix, test

# Define Flower Client and client_fn
class XgbClient(Client):
    def __init__(
        self,
        train_dmatrix : xgb.DMatrix,
        valid_dmatrix : xgb.DMatrix,
        num_local_round : int,
        xgb_params: dict,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_local_round = num_local_round
        self.xgb_params = replace_keys(xgb_params)

    def _local_boost(self, bst_input: xgb.Booster):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for server aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds() - self.num_local_round
            : bst_input.num_boosted_rounds()
        ]
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["server_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.xgb_params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.xgb_params)
            global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)
        
        train_labels = self.train_dmatrix.get_label()
        training_points = len(train_labels)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=training_points,
            metrics={},
        )
        
    def decode_eval_results(self, eval_results):
        res_dict = {}
        parts = eval_results.split("\t")
        for eval_result in parts:
            if ":" in eval_result:
                split = eval_result.split(":")
                res_dict[split[0]] = round(float(split[1]), 4)
        return res_dict
    

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.xgb_params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        eval_results = self.decode_eval_results(eval_results)
        auc = eval_results["valid-auc"]
        
        accuracy, metrics = test(bst, self.valid_dmatrix)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=len(self.valid_dmatrix.get_label()),
            metrics={
                "accuracy": accuracy,
                "auc": auc,
                "metrics": metrics,
            },
        )
        
# Flower Client - standard-FL XGBoost
def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_local_round = context.run_config["local-epochs"]
    
    train_dmatrix, valid_dmatrix = load_data_dmatrix(
        partition_id, num_partitions, context.run_config
    )

    hyperparams = get_hyperparams_from_config(context.run_config, model="xgb")
    
    # Return Client instance
    return XgbClient(
        train_dmatrix,
        valid_dmatrix,
        num_local_round,
        hyperparams,
    )


# Flower ClientApp
app = ClientApp(
    client_fn,
)