from torch.utils.data import DataLoader
from torch import nn
import torch


def calculate_fisher_information(model: torch.nn.Module, data_loader: DataLoader):
    """Calculate Fisher information for EWC"""
    model.eval()  # Set to evaluation mode
    
    # Initialize Fisher information
    fisher_diag = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    
    total_samples = 0
    
    # Calculate Fisher information
    for batch in data_loader:
        data = batch["features"]
        labels = batch["labels"]
        batch_size = data.size(0)
        total_samples += batch_size
        
        # Ensure labels are long type
        labels = labels.long()
        
        
        # Forward pass
        model.zero_grad()
        output: torch.tensor = model(data)
        # print(f"Output shape: {output.shape}, output: {output}")
        singular_output = output[0][0].data
        # print(f"Singular output shape: {singular_output.shape}, singular output: {singular_output}")
        
        pos_prob = torch.sigmoid(output)
        log_likelihood = 0
        for i in range(batch_size):
            if labels[i] == 1:
                log_likelihood += torch.log(pos_prob[i])
            else:
                log_likelihood += torch.log(1 - pos_prob[i])
        
        # Negate because we need to maximize log likelihood
        (-log_likelihood).backward()
        
        # Accumulate squared gradients (diagonal Fisher)
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_diag[name] += singular_output * param.grad.pow(2).data
    
    # Normalize by the number of samples
    for name in fisher_diag:
        fisher_diag[name] /= total_samples
        
    return fisher_diag