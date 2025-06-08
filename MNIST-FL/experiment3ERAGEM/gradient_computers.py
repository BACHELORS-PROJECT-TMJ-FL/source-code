import torch
import numpy as np

def flatten_gradient(gradient: list[torch.tensor]) -> np.array:
    return np.concatenate([g.numpy().flatten() for g in gradient], axis=0)

def restructure_gradient(gradient: np.array, shape: list[tuple[int]]) -> list[torch.tensor]:
    restructured_gradient = []
    for ind_shape in shape:
        # Get gradients of first layer
        temp_grad = gradient[:np.prod(ind_shape)]
        # Remove used gradients
        gradient = gradient[np.prod(ind_shape):]
        # Reshape first gradient
        temp_grad = temp_grad.reshape(ind_shape)
        restructured_gradient.append(torch.tensor(temp_grad))
    return restructured_gradient
    


def fedGEMGradCalc(gradients: list[list[torch.tensor]], main_id: int = 0) -> list[torch.tensor]:
    """
    Calculate the gradient for the main client using the gradients from all clients.
    The main client is the one that will be used to update the model.
    """
    # Initialize the gradient for the main client
    main_client_gradient = gradients[main_id].copy()
    other_client_gradients = gradients[:main_id] + gradients[main_id + 1:]
    
    # Flatten gradients for calculating cos(a)
    main_client_gradient_flat = flatten_gradient(main_client_gradient)
    other_client_gradients_flat = [flatten_gradient(grad) for grad in other_client_gradients]
    
    main_client_gradient_altered_flat = main_client_gradient_flat.copy()
    for other_gradient in other_client_gradients_flat:
        # Calculate the cosine similarity between the main client gradient and the other client gradient
        cos_similarity = np.dot(main_client_gradient_flat, other_gradient) / (np.linalg.norm(main_client_gradient_flat) * np.linalg.norm(other_gradient))
        
        # If cos_similarity is negattive, we want to subtract the projection of the main gradient on the other gradient from the main gradient
        if (cos_similarity < 0):
            # Calculate the projection of the main gradient on the other gradient
            projection = (np.dot(main_client_gradient_altered_flat, other_gradient) / np.linalg.norm(other_gradient)**2) * other_gradient
            # Subtract the projection from the main gradient
            main_client_gradient_flat -= projection
    
    # Recreate flattened gradient
    main_client_gradient_restructured = restructure_gradient(main_client_gradient_altered_flat, [grad.shape for grad in main_client_gradient])
    
    return main_client_gradient_restructured

def fedAGEMGradCalc(gradients: list[list[torch.tensor]], main_id: int = 0) -> list[torch.tensor]:
    """
    Calculate the gradient for the main client using the gradients from all clients.
    The main client is the one that will be used to update the model.
    """
    # Initialize the gradient for the main client
    main_client_gradient = gradients[main_id].copy()
    other_client_gradients = gradients[:main_id] + gradients[main_id + 1:]
    
    # Flatten gradients for calculating cos(a)
    main_client_gradient_flat: np.ndarray = flatten_gradient(main_client_gradient)
    other_client_gradients_flat: list[np.ndarray] = [flatten_gradient(grad) for grad in other_client_gradients]

    other_client_gradients_mean_flat = np.mean(other_client_gradients_flat, axis=0)
    
    main_client_gradient_altered_flat = main_client_gradient_flat.copy()
    
    # Calculate the cosine similarity between the main client gradient and the average gradient
    cos_similarity = np.dot(main_client_gradient_flat, other_client_gradients_mean_flat) / (np.linalg.norm(main_client_gradient_flat) * np.linalg.norm(other_client_gradients_mean_flat))
    
    # If cos_similarity is negattive, we want to subtract the projection of the main gradient on the other gradient from the main gradient
    if (cos_similarity < 0):
        # Calculate the projection of the main gradient on the other gradient
        projection = (np.dot(main_client_gradient_altered_flat, other_client_gradients_mean_flat) / np.linalg.norm(other_client_gradients_mean_flat)**2) * other_client_gradients_mean_flat
        # Subtract the projection from the main gradient
        main_client_gradient_flat -= projection
    
    # Recreate flattened gradient
    main_client_gradient_restructured = restructure_gradient(main_client_gradient_altered_flat, [grad.shape for grad in main_client_gradient])
    
    return main_client_gradient_restructured