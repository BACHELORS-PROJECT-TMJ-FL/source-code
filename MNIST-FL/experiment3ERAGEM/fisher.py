from torch.utils.data import DataLoader
from torch import nn
import torch
def calculate_fisher_information_domain(model: torch.nn.Module, data_loader: DataLoader, base_criterion: torch.nn.Module):
    """Calculate Fisher information for EWC"""
    model.eval()  # Set to evaluation mode
    
    # Initialize Fisher information
    fisher_diag = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    
    # Calculate Fisher information
    for batch in data_loader:
        data = batch["image"]
        labels = torch.tensor(batch["label"]).type(torch.float32) % 2
        
        # Forward pass
        model.zero_grad()
        output = model(data)[:, 0]
        
        # Calculate loss (using log likelihood for proper Fisher)
        loss: torch.Tensor = base_criterion(output, labels)
        loss.backward()

        singular_output = output[0]

        # Accumulate squared gradients (diagonal Fisher)
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_diag[name] += singular_output.data * param.grad.pow(2).data
    
    # Normalize by the number of samples
    for name in fisher_diag:
        fisher_diag[name] /= len(data_loader)
        
    # Update the EWC loss with new importance weights
    return fisher_diag

import numpy as np

def calculate_fisher_information_task(model: torch.nn.Module, data_loader: DataLoader, base_criterion: torch.nn.Module):
    """Calculate Fisher information for EWC"""
    model.eval()  # Set to evaluation mode
    
    # Initialize Fisher information
    fisher_diag = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    
    # Calculate Fisher information
    for batch in data_loader:
        data = batch["image"]
        labels = batch["label"]
        # convert labels to 5d vector with 0, 1 being the first index
        labels_handled = np.zeros((len(labels), 5), dtype=np.float32)
        indexes = []
        for i in range(len(labels)):
            index = labels[i] // 2
            labels_handled[i][index] = labels[i] % 2
            indexes.append(index)
        labels = torch.tensor(labels_handled)
        # Forward pass
        model.zero_grad()
        output = model(data)
        labels = labels.cpu().type(torch.float32)

        # Calculate loss (using log likelihood for proper Fisher)
        loss: torch.Tensor = base_criterion(output, labels)
        loss.backward()
        
        single_output = output[:, indexes[0]]  # Assuming we want the first output for Fisher calculation

        # Accumulate squared gradients (diagonal Fisher)
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_diag[name] += single_output.data * param.grad.pow(2).data
    
    # Normalize by the number of samples
    for name in fisher_diag:
        fisher_diag[name] /= len(data_loader)
        
    # Update the EWC loss with new importance weights
    return fisher_diag