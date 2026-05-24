import xarray as xr
import torch
import numpy as np
from torch.utils.data import Dataset 
# FIX: Import DataLoader from PyTorch Geometric, not standard PyTorch
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from torch_geometric.data import Data

class WeatherDataset(Dataset):
    def __init__(self, zarr_path, graph_path):
        self.ds = xr.open_zarr(zarr_path)
        self.graph = torch.load(graph_path, weights_only=False)
        self.time_steps = len(self.ds.time)

        # Load the global statistics
        self.global_mean = np.load("data/processed/train_mean.npy")
        self.global_std = np.load("data/processed/train_std.npy")

    def __len__(self):
        # We need at least 2 consecutive steps for t -> t+1
        return self.time_steps - 1

    def __getitem__(self, idx):
        # 1. Select the time slices
        # Current shape of ds: (time, isobaricInhPa, node) for each variable
        snap_t = self.ds.isel(time=idx)
        snap_t_plus_1 = self.ds.isel(time=idx+1)
        
        def process_snapshot(snap):
            # Convert variables to a single dimension
            # Resulting shape: (variable, isobaricInhPa, node)
            arr = snap.to_array(dim='variable')
            
            # Stack 'variable' and 'isobaricInhPa' into a single 'feature' dimension
            # and ensure 'node' is the first dimension
            # Resulting shape: (node, feature)
            arr = arr.stack(features=['variable', 'isobaricInhPa']).transpose('node', 'features')
            
            val = arr.values
            # Apply the GLOBAL normalization
            val_normalized = (val - self.global_mean) / (self.global_std + 1e-6)
            return torch.tensor(val_normalized, dtype=torch.float32)

        x = process_snapshot(snap_t)
        y = process_snapshot(snap_t_plus_1)
        
        return Data(
            x=x,
            y=y,
            edge_index=self.graph.edge_index,
            edge_attr=self.graph.edge_attr
        )

class AfriCastDataModule(pl.LightningDataModule):
    def __init__(self, zarr_path, graph_path, batch_size=1):
        super().__init__()
        self.zarr_path = zarr_path
        self.graph_path = graph_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = WeatherDataset(self.zarr_path, self.graph_path)

    def train_dataloader(self):
        # This now uses the PyG DataLoader which handles the collation of Data objects automatically
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True # Speeds up data transfer
        )