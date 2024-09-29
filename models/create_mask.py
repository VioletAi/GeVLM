import torch

def calculate_distances(locations):
    #calculate pairwise Euclidean distances between objects
    
    obj1 = locations.unsqueeze(2)
    obj2 = locations.unsqueeze(1)
    
    distances = torch.sqrt(torch.sum((obj1 - obj2) ** 2, dim=-1))
    
    return distances

def create_object_specific_mask(locations, percentile, num_heads):
    distances = calculate_distances(locations)
    # Calculate the percentile distance for each object in the batch
    percentiles = torch.quantile(distances, percentile, dim=2, keepdim=True)
    # Create a mask where positions beyond each object's percentile are set to -inf
    mask = torch.where(distances <= percentiles, torch.tensor(0.0, device=distances.device), torch.tensor(float('-inf'), device=distances.device))
    # Adjust mask dimensions for MultiheadAttention
    batch_size, seq_len = distances.size(0), distances.size(1)
    mask = mask.unsqueeze(1).repeat(1, num_heads, 1, 1)  # Duplicate for each head
    mask = mask.view(batch_size * num_heads, seq_len, seq_len)  # Reshape for multihead attention
    return mask