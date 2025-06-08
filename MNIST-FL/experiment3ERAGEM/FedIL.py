from flwr.server.strategy import FedAvg
from flwr.common import FitIns
from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy

class NaiveFILCriterion(Criterion):
    def __init__(self, available_cids: list[str], server_round: int, rounds_per_client: int = 2):
        super().__init__()
        self.server_round = server_round
        self.rounds_per_client = rounds_per_client
        # Sort the cids
        self.cids_sorted = sorted([int(cid) for cid in available_cids])

    def select(self, client: ClientProxy):
        client_index = (self.server_round - 1) // self.rounds_per_client
        # Select the clients with the lowest cids
        selected_clients = [self.cids_sorted[client_index]]
        return int(client.cid) in selected_clients

        

class NaiveFIL(FedAvg):
    def __init__(self, context=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = context

    def configure_fit(self, server_round, parameters, client_manager): 
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        # sample_size, min_num_clients = self.num_fit_clients(
        #     client_manager.num_available()
        # )

        available_cids = list(client_manager.clients)
        criterion = NaiveFILCriterion(available_cids, server_round, rounds_per_client=self.context.run_config["rounds-per-client"])

        clients = client_manager.sample(
            num_clients=1, min_num_clients=1, criterion=criterion 
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
    def num_evaluation_clients(self, num_available_clients):
        return self.min_evaluate_clients, self.min_evaluate_clients # Must use all clients for evaluation
  
class NaiveReplayFILCriterion(Criterion):
    def __init__(self, available_cids: list[str], server_round: int, rounds_per_client: int = 2):
        super().__init__()
        self.server_round = server_round
        self.rounds_per_client = rounds_per_client
        # Sort the cids
        self.cids_sorted = sorted([int(cid) for cid in available_cids])

    def select(self, client: ClientProxy):
        num_clients_needed = (self.server_round - 1) // self.rounds_per_client + 1
        # Select the clients with the lowest cids
        selected_clients = self.cids_sorted[:num_clients_needed]
        return int(client.cid) in selected_clients

        

class NaiveReplayFIL(FedAvg):
    def __init__(self, context=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = context

    def configure_fit(self, server_round, parameters, client_manager): 
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        # sample_size, min_num_clients = self.num_fit_clients(
        #     client_manager.num_available()
        # )

        available_cids = list(client_manager.clients)
        criterion = NaiveReplayFILCriterion(available_cids, server_round, rounds_per_client=self.context.run_config["rounds-per-client"])

        num_clients_needed = (server_round - 1) // self.context.run_config["rounds-per-client"] + 1
        clients = client_manager.sample(
            num_clients=num_clients_needed, min_num_clients=1, criterion=criterion 
        
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
    def num_evaluation_clients(self, num_available_clients):
        return self.min_evaluate_clients, self.min_evaluate_clients # Must use all clients for evaluation  
    
class StandardFL(FedAvg):
    def __init__(self, context=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_fit(self, server_round, parameters, client_manager): 
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]


def get_FILStrategy(strategy: str):
    """Get the strategy based on the ILVariant."""
    if strategy == "NaiveFIL":
        return NaiveFIL
    elif strategy == "NaiveReplayFIL":
        return NaiveReplayFIL
    elif strategy == "StandardFL":
        return StandardFL
    else:
        raise ValueError(f"Unknown strategy: {strategy}")