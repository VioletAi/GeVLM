import torch
import os

# Define the folder path where your .pt files are stored
folder_path = "distance_tensor_per_scene/"

# List all files in the folder
pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

# Iterate through all .pt files and check the range of the tensor
for pt_file in pt_files:
    file_path = os.path.join(folder_path, pt_file)
    
    # Load the tensor from the .pt file
    tensor = torch.load(file_path)
    
    # Check the minimum and maximum values
    tensor_min = tensor.min().item()
    tensor_max = tensor.max().item()
    
    # Print out the range
    print(f"{pt_file}: min = {tensor_min}, max = {tensor_max}")
