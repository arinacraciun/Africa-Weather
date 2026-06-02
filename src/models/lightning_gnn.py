import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv

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
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate=1e-4, physics_lambda=0.1):
        super().__init__()
        self.gnn = AIFSGNN(in_channels, hidden_channels, out_channels)
        self.lr = learning_rate
        self.loss_fn = nn.MSELoss()
        
        # Lambda controls how strictly we enforce the physical rules vs fitting the data
        self.physics_lambda = physics_lambda

    def forward(self, batch):
        # 1. Calculate the residual (how the weather changes)
        residual = self.gnn(batch.x, batch.edge_index)
        
        # 2. Add the residual to the current state (batch.x)
        # This is the Residual Connection that prevents exponential explosion
        next_state_prediction = batch.x + residual

        return next_state_prediction
    
    def calculate_physics_loss(self, predictions, edge_index):
        """
        Calculates the spatial gradient across the graph. 
        Penalizes physically impossible sharp spikes between neighboring nodes.
        """
        # Extract source and destination node indices from the graph edges
        src, dst = edge_index
        
        # Calculate the squared difference in weather variables between all connected nodes
        node_differences = (predictions[src] - predictions[dst]) ** 2
        
        # The mean of these differences acts as our smoothness penalty
        spatial_loss = node_differences.mean()
        
        return spatial_loss

    def training_step(self, batch, batch_idx):
        # 1. Forward Pass (Predict state t+1)
        predictions = self(batch)
        
        # 2. Standard Data Loss (How close are we to the true ERA5 data?)
        mse_loss = self.loss_fn(predictions, batch.y)
        
        # 3. Physics Loss (Are we breaking the laws of thermodynamics?)
        phys_loss = self.calculate_physics_loss(predictions, batch.edge_index)
        
        # 4. Total Loss Calculation
        total_loss = mse_loss + (self.physics_lambda * phys_loss)
        
        # Log all three metrics so you can watch them individually during training
        self.log('train_loss', total_loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log('mse_loss', mse_loss, batch_size=batch.num_graphs)
        self.log('physics_loss', phys_loss, batch_size=batch.num_graphs)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)