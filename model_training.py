"""
Model Training Pipeline

This script defines the Graph Neural Network architecture, handles data loading 
via PyTorch Geometric, and orchestrates the training loop using PyTorch Lightning.
It includes physics-informed loss functions to penalize physically impossible predictions.
"""

import os
import numpy as np
import xarray as xr

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# PyTorch Geometric specific imports
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

# =============================================================================
# 1. DATASET & DATAMODULE
# =============================================================================

class WeatherDataset(Dataset):
    """
    Custom Dataset to load Zarr data lazily and format it as PyTorch Geometric graphs.
    """
    def __init__(self, zarr_path, graph_path, stats_dir):
        self.ds = xr.open_zarr(zarr_path)
        self.graph = torch.load(graph_path, weights_only=False)
        self.time_steps = len(self.ds.time)

        # Load global statistics for normalization
        self.global_mean = np.load(os.path.join(stats_dir, "train_mean.npy"))
        self.global_std = np.load(os.path.join(stats_dir, "train_std.npy"))

    def __len__(self):
        # Requires at least 2 consecutive steps for autoregressive (t -> t+1) mapping
        return self.time_steps - 1

    def __getitem__(self, idx):
        # Select time slices
        snap_t = self.ds.isel(time=idx)
        snap_t_plus_1 = self.ds.isel(time=idx+1)
        
        def process_snapshot(snap):
            # Convert to DataArray and stack into (node, features)
            arr = snap.to_array(dim='variable')
            arr = arr.stack(features=['variable', 'isobaricInhPa']).transpose('node', 'features')
            
            val = arr.values
            # Apply global normalization
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
    """
    PyTorch Lightning DataModule to handle data loading configuration.
    """
    def __init__(self, zarr_path, graph_path, stats_dir, batch_size=1):
        super().__init__()
        self.zarr_path = zarr_path
        self.graph_path = graph_path
        self.stats_dir = stats_dir
        self.batch_size = batch_size
        self.dataset = None

    def setup(self, stage=None):
        self.dataset = WeatherDataset(self.zarr_path, self.graph_path, self.stats_dir)

    def train_dataloader(self):
        # Uses PyG DataLoader to handle collating graph Data objects automatically
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True
        )

# =============================================================================
# 2. MODEL ARCHITECTURE
# =============================================================================

class AIFSGNN(nn.Module):
    """
    Core Graph Neural Network defining the forward propagation.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, out_features)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class WeatherLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper handling the training step, loss calculations, 
    and residual connection.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate=1e-4, physics_lambda=0.1):
        super().__init__()
        self.gnn = AIFSGNN(in_channels, hidden_channels, out_channels)
        self.lr = learning_rate
        self.loss_fn = nn.MSELoss()
        self.physics_lambda = physics_lambda

    def forward(self, batch):
        # Calculate residual (the delta of how the weather changes)
        residual = self.gnn(batch.x, batch.edge_index)
        
        # Add residual to current state (prevents exponential explosion)
        next_state_prediction = batch.x + residual
        return next_state_prediction
    
    def calculate_physics_loss(self, predictions, edge_index):
        """
        Penalizes physically impossible sharp spikes between neighboring nodes.
        """
        src, dst = edge_index
        node_differences = (predictions[src] - predictions[dst]) ** 2
        spatial_loss = node_differences.mean()
        return spatial_loss

    def training_step(self, batch, batch_idx):
        predictions = self(batch)
        
        mse_loss = self.loss_fn(predictions, batch.y)
        phys_loss = self.calculate_physics_loss(predictions, batch.edge_index)
        
        total_loss = mse_loss + (self.physics_lambda * phys_loss)
        
        self.log('train_loss', total_loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log('mse_loss', mse_loss, batch_size=batch.num_graphs)
        self.log('physics_loss', phys_loss, batch_size=batch.num_graphs)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# =============================================================================
# 3. TRAINING EXECUTION
# =============================================================================

def run_training_pipeline():
    """
    Sets up the data, initializes the model, configures callbacks, 
    and starts the training loop.
    """
    # Define paths (Assumes data_preparation.py has been run)
    processed_dir = "data/processed"
    zarr_path = os.path.join(processed_dir, "era5_2023_01.zarr")
    graph_path = os.path.join(processed_dir, "static_graph.pt")
    
    # Check if data exists to prevent silent failures
    if not os.path.exists(zarr_path) or not os.path.exists(graph_path):
        raise FileNotFoundError(
            "Processed data not found. Please run the data preparation pipeline first."
        )

    # 1. Initialize DataModule
    print("Setting up DataModule...")
    dm = AfriCastDataModule(
        zarr_path=zarr_path, 
        graph_path=graph_path, 
        stats_dir=processed_dir,
        batch_size=1
    )
    dm.setup()
    
    # 2. Initialize Model
    print("Initializing Model...")
    num_features = 5 * 6 # 5 variables, 6 pressure levels
    model = WeatherLightningModule(
        in_channels=num_features, 
        hidden_channels=64, 
        out_channels=num_features
    )
    
    # 3. Configure Trainer
    checkpoint_callback = ModelCheckpoint(monitor='train_loss', mode='min')
    early_stop = EarlyStopping(monitor="train_loss", patience=3, mode="min")
    
    trainer = pl.Trainer(
        max_epochs=15,
        accelerator="cpu",
        precision="bf16-mixed", 
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=1
    )
    
    # 4. Start Training
    print("Starting Training Loop...")
    trainer.fit(model, datamodule=dm)
    print("Training complete!")

if __name__ == "__main__":
    run_training_pipeline()