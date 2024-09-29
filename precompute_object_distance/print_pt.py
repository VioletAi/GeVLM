import torch
import os
import math

def compute_distance(p1, p2):
    # Compute Euclidean distance between two points p1 and p2
    return math.sqrt((p1[0] - p2[0]) ** 2 +
                     (p1[1] - p2[1]) ** 2 +
                     (p1[2] - p2[2]) ** 2)

def process_scenes(file_path, output_folder):
    # Load the data from the file
    data = torch.load(file_path)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each scene
    for scene_id, scene_data in data.items():
        # Extract locations (bounding box centers) from the scene
        locs = scene_data['locs']  #[x, y, z, w, h, l]
        
        # Number of objects in the scene
        num_objects = len(locs)

        # Initialize a num_objects x num_objects tensor to store distances
        distance_matrix = torch.zeros((num_objects, num_objects))

        # Calculate pairwise distances
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    # Get center points [x, y, z] for object i and object j
                    p1 = locs[i][:3]  # First 3 entries are [x, y, z]
                    p2 = locs[j][:3]
                    # Compute the distance
                    distance_matrix[i, j] = compute_distance(p1, p2)

        # Normalize the distance matrix to range [0, 1], excluding zeros
        non_zero_mask = distance_matrix != 0
        non_zero_distances = distance_matrix[non_zero_mask]
        if non_zero_distances.numel() > 0:
            min_distance = non_zero_distances.min()
            max_distance = non_zero_distances.max()
            if max_distance > min_distance:
                # Normalize only the non-zero entries
                distance_matrix[non_zero_mask] = (distance_matrix[non_zero_mask] - min_distance) / (max_distance - min_distance)
                # The zeros remain zeros
            else:
                # If all distances are equal, set the non-zero entries to zero
                distance_matrix[non_zero_mask] = 0
        else:
            # If there are no non-zero distances, keep the matrix as zeros
            pass  # distance_matrix is already zeros

        # Save the normalized distance matrix tensor as 'scene_id.pt'
        output_path = os.path.join(output_folder, f"{scene_id}.pt")
        torch.save(distance_matrix, output_path)
        print(f"Saved normalized distance matrix for {scene_id} at {output_path}")

# Usage example
file_path = 'annotations/scannet_mask3d_train_attributes.pt'  # Replace with the actual file path
output_folder = 'distance_tensor_per_scene/'  # Folder to save the results
process_scenes(file_path, output_folder)
