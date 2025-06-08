"""xgboost_quickstart: A Flower / XGBoost app."""

import xgboost as xgb
from common.dental_data import load_client_data
from common.data import make_dmatrix
from common.binary_classification import TestResultBinClassifcation

from flwr.common import Parameters, Metrics

from flwr.server.strategy import FedXgbBagging
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, FitRes,  Parameters

from common import report as r

from common.save_model import save_xgb_parameters
 

# Client
def load_data_dmatrix(partition_id: int, num_partitions: int, config) :
    """Get dataset in DMatrix format. for the particular partition_id."""
    
    trainset, testset = load_client_data(partition_id, num_partitions, config)
    
    train_dmatrix = make_dmatrix(trainset)
    valid_dmatrix = make_dmatrix(testset)
    
    return train_dmatrix, valid_dmatrix


def test(bst: xgb.Booster, test_dmatrix: xgb.DMatrix) -> tuple[float, str]:
    """
    Test the xgboost model on the test_dmatrix.
    returns a tuple: (accuracy, metrics(json))
    """
    
    preds = bst.predict(test_dmatrix)
    
    labels = test_dmatrix.get_label()
    
    tr = TestResultBinClassifcation()
    tr.compare_batch(preds, labels)
    
    return tr.accuracy(), tr.to_json()

import json

def get_model_info(bst: xgb.Booster) -> dict:
    
    # Get params of the model
    params = bst.save_config()
    params = json.loads(params)
    
    gb_params = params["learner"]["gradient_booster"] 
    num_trees = gb_params["gbtree_model_param"]["num_trees"]
    eta = gb_params["tree_train_param"]["eta"]
    
    
    return {
        "num_trees": num_trees,
        "eta": eta,
    }
    
class CustomFedXbg(FedXgbBagging):
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
        
        
        if self.save_model and self.save_at_round > 0 and server_round == self.save_at_round:
            if parameters_aggregated is not None:
                save_xgb_parameters(
                    parameters_aggregated, self.fold
                )
                print(f"Server round {server_round}: Model parameters saved.")
            else:
                print(f"Server round {server_round}: No parameters to save.")

        return parameters_aggregated, metrics_aggregated
        
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, Metrics]:
        loss, metrics = super().evaluate(server_round, parameters)
        if not (metrics is {}):            
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
    
    def initialize_parameters(self, client_manager):
        if len(self.initial_parameters.tensors) > 0:
            self.global_model = self.initial_parameters.tensors[0]
        return super().initialize_parameters(client_manager)
    