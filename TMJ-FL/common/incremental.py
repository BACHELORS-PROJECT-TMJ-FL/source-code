
import torch
import torch.nn as nn
import numpy as np
from common.config import get_il_config
from torch.utils.data import DataLoader

from common.fisher import calculate_fisher_information

def total_il_server_rounds(run_config) -> int:
    """
    Calculate the total number of rounds for incremental learning.
    """
    il_config = get_il_config(run_config)

    initial_server_rounds = il_config["num-initial-server-rounds"]
    
    if il_config["stop-after-initial"]:
        # If we stop after initial rounds, return that number
        return initial_server_rounds
    
    rounds_per_increment = il_config["num-rounds-per-increment"]
    increments = il_config["num-increments"]
    
    

    return initial_server_rounds + rounds_per_increment * increments


class IncrementalStrategy():
    def __init__(self, run_config):
        
        il_config = get_il_config(run_config)
        self.il_config = il_config
        self.current_criterion_stage = 0
        
        if il_config["data-strategy"] == "buffer":
            self.data_strat = BufferDataStrategy(self.il_config)
        elif il_config["data-strategy"] == "replay":
            self.data_strat = ReplayDataStrategy(self.il_config)
        
        self.use_ewc = "ewc-lambda" in il_config and il_config["ewc-lambda"] > 0
        if self.use_ewc:
            lambda_value = il_config["ewc-lambda"]
            self.base_criterion = torch.nn.BCELoss()
            self.incremental_criterion = EWCLoss(
                self.base_criterion,
                importance_weights={},
                lamb=lambda_value
            )
            print("Using EWC with lambda:", lambda_value)
        else:
            self.incremental_criterion = torch.nn.BCELoss()
            print("Not using EWC")

    def get_criterion(self, model: torch.nn.Module, data: DataLoader, stage: int) -> torch.nn.Module:
        if not self.use_ewc or stage == 0:
            return torch.nn.BCELoss()
        else:   
            if self.current_criterion_stage != stage:
                fisher_diag = calculate_fisher_information(model, data)
                self.incremental_criterion.update_model_importance(model, fisher_diag)
                self.current_criterion_stage = stage
            
            return self.incremental_criterion

    def get_il_stage(self, global_round: int) -> int:
        initial_server_rounds = self.il_config["num-initial-server-rounds"]
        rounds_per_increment = self.il_config["num-rounds-per-increment"]
        
        # Check if we're still in the initial stage
        if global_round <= initial_server_rounds:
            return 0
        else:
            incremental_stage = ((global_round - initial_server_rounds - 1) // rounds_per_increment) + 1
                
            return incremental_stage
        
    
    def get_il_metrics(self, global_round: int) -> dict:

        return {"stage": self.get_il_stage(global_round)}

    def data_until_stage(self, num_points: int, stage: int) -> np.ndarray:
        """
        Get data indices until the specified stage.
        """
        initial_upper = self.data_strat.get_initial_upper_bounds(num_points)
        stage_before = stage - 1

        if stage_before == 0:
            return np.arange(initial_upper)
        else:
            _, upper_bound = self.data_strat.get_stage_bounds(num_points, stage_before, initial_upper)
            return np.arange(upper_bound)
    
    def sample_data(self, num_points: int, global_round: int) -> np.ndarray:

        return self.data_strat.sample_data(num_points, self.get_il_stage(global_round))



class ILDataStrategy():
    def __init__(self, il_config):
        self.il_config = il_config

    def sample_data(self, num_points: int, stage: int) -> np.ndarray:
        """
        Sample data for the current stage of incremental learning.
        """        
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def get_initial_upper_bounds(self, num_points: int) -> tuple:
        """
        Get the initial bounds for the data sampling.
        """
        initial_round_percentage = self.il_config["initial-round-percentage"]
        initial_upper = int(num_points * initial_round_percentage)
        return initial_upper

    def get_stage_bounds(self, num_points: int, stage: int, initial_upper: int) -> tuple:
        """
        Get the bounds for the current stage of incremental learning.
        """
        num_increments = self.il_config["num-increments"]

        remaining_points = num_points - initial_upper
        increment_size = remaining_points // num_increments

        lower_bound = initial_upper + int((stage-1) * increment_size)
        upper_bound = min(lower_bound + increment_size, num_points)
        
        return lower_bound, upper_bound
    
class ReplayDataStrategy(ILDataStrategy):
    def __init__(self, il_config):
        super().__init__(il_config)
        
        self.replay_percentage = il_config["replay-percentage"]

    def sample_data(self, num_points: int, stage: int) -> np.ndarray:

        initial_upper = self.get_initial_upper_bounds(num_points)

        if stage == 0:
            return np.arange(initial_upper)
        else:
            lower_bound, upper_bound = self.get_stage_bounds(num_points, stage, initial_upper)

            new_data_indices = np.arange(lower_bound, upper_bound)

            # In replay strategy, we always sample the first part of the data
            # as a percentage of the data before the current stage

            num_replay_samples = max(0, int(lower_bound * self.replay_percentage))

            replay_indices = np.random.choice(
                np.arange(lower_bound), 
                num_replay_samples, 
                replace=False
            )

            combined_indices = np.concatenate((replay_indices, new_data_indices))

            return combined_indices
        
class BufferDataStrategy(ILDataStrategy):
    def __init__(self, il_config):
        super().__init__(il_config)
        
        self.buffer_budget = il_config["buffer-budget"]
        self.buffer = []
        self.buffer_stage = -1

    def sample_data(self, num_points: int, stage: int) -> np.ndarray:
        # First get initial upper bounds
        initial_upper = self.get_initial_upper_bounds(num_points)
        
        if stage == 0:
            data_indices = np.arange(initial_upper)
            
            # Update buffer if we've moved to a new stage
            if stage > self.buffer_stage:
                self.update_buffer(data_indices)
                self.buffer_stage = stage
            
            return data_indices
        
        else:
            lower_bound, upper_bound = self.get_stage_bounds(num_points, stage, initial_upper)
            new_data_indices = np.arange(lower_bound, upper_bound)
            
            # Update buffer if we've moved to a new stage
            if stage > self.buffer_stage:
                self.update_buffer(new_data_indices)
                self.buffer_stage = stage
            
            # Combine buffer with new data 
            combined_indices = np.concatenate((self.buffer, new_data_indices))

            return combined_indices

    def update_buffer(self, new_data_indices):
        if len(self.buffer) >= self.buffer_budget:
            # If buffer is full, remove some old data
            max_in_buffer = max(self.buffer) if self.buffer else 0
            max_in_new = max(new_data_indices) if len(new_data_indices) > 0 else 0
            
            if max_in_buffer > 0:  # Protect against division by zero
                percentage_to_remove = min(0.9, (max_in_new / max_in_buffer))  # Cap at 90%
                num_to_remove = int(percentage_to_remove * len(self.buffer))
                
                if num_to_remove > 0:
                    indices_to_remove = np.random.choice(
                        len(self.buffer),  # Choose indices in the buffer array
                        num_to_remove, 
                        replace=False
                    )
                    # Create a new buffer without the removed indices
                    self.buffer = [self.buffer[i] for i in range(len(self.buffer)) 
                                  if i not in indices_to_remove]
        
        # Calculate how many new elements to add
        num_to_add = min(
            len(new_data_indices),  # Don't try to add more than available
            self.buffer_budget - len(self.buffer)  # Don't exceed budget
        )
        
        if num_to_add > 0 and len(new_data_indices) > 0:
            # Sample without replacement from new data
            new_buffer_indices = np.random.choice(
                new_data_indices,
                num_to_add,
                replace=False
            )
            
            # Extend buffer with new samples
            self.buffer.extend(new_buffer_indices)
        
        # Verify buffer size doesn't exceed budget
        if len(self.buffer) > self.buffer_budget:
            # If we somehow have too many elements, trim
            self.buffer = self.buffer[:self.buffer_budget]
        
        # Correct assertion
        assert len(self.buffer) <= self.buffer_budget, "Buffer size exceeds budget"
 

# EWC Loss function
# https://arxiv.org/pdf/1612.00796
class EWCLoss(nn.Module):
    def __init__(self, base_criterion, importance_weights, lamb=1.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.importance_weights = importance_weights  
        self.old_params = None  
        self.lamb = lamb
        self.current_model: nn.Module = None
        
    def update_model_importance(self, model, fisher_diag):
        """Update importance weights based on Fisher information"""
        print("Updating importance weights")
        self.importance_weights = fisher_diag
        print(self.importance_weights)
        self.old_params = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.current_model = model
        
    def forward(self, output, target):
        # Standard task loss
        loss = self.base_criterion(output, target)
        
        if self.old_params is None or self.importance_weights is None:
            return loss

        # Add EWC penalty
        ewc_loss = 0
        for name, param in self.current_model.named_parameters():
            if name in self.old_params and name in self.importance_weights:
                ewc_loss += (self.importance_weights[name] * 
                            (param - self.old_params[name]).pow(2)).sum()


        print("EWC loss:", ewc_loss.item(), "After lambda: ", self.lamb * ewc_loss.item(), "\nLoss:", loss.item())

        return loss + self.lamb * ewc_loss