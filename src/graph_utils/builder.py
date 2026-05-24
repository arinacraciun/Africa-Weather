import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

def create_spherical_graph(latitudes, longitudes, k_neighbors=6):
    """
    Constructs a K-Nearest Neighbors graph over the surface of a sphere.
    """
    R = 6371.0 # Earth radius in km
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    
    # Convert to 3D Cartesian coordinates
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    
    coords = np.stack([x, y, z], axis=1)
    
    # Calculate connections using Euclidean distance in 3D space
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Establish source and target nodes for PyG edge_index
    # Index 0 is the node itself, so we slice from 1 onwards
    src = np.repeat(np.arange(len(coords)), k_neighbors)
    dst = indices[:, 1:].flatten()
    
    # PyTorch Geometric format
    edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
    edge_attr = torch.tensor(distances[:, 1:].flatten(), dtype=torch.float32).unsqueeze(1)
    
    return Data(edge_index=edge_index, edge_attr=edge_attr, pos=torch.tensor(coords, dtype=torch.float32))