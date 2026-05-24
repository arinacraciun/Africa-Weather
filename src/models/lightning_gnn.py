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
    """
    Lightning Module managing the training step, loss, and optimizers.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate=1e-4):
        super().__init__()
        self.gnn = AIFSGNN(in_channels, hidden_channels, out_channels)
        self.lr = learning_rate
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        # batch is a PyTorch Geometric DataBatch object
        return self.gnn(batch.x, batch.edge_index)

    def training_step(self, batch, batch_idx):
        # Predict state t+1
        predictions = self(batch)
        
        # batch.y contains the target ground truth at t+1
        loss = self.loss_fn(predictions, batch.y)
        
        # Lightning handles logging automatically
        self.log('train_loss', loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)