import torch
import xarray as xr
import numpy as np
from torch_geometric.data import Data
from src.models.lightning_gnn import WeatherLightningModule

def load_model_and_graph(checkpoint_path, graph_path):
    # 1. Load the model from the saved checkpoint
    model = WeatherLightningModule.load_from_checkpoint(
        checkpoint_path,
        in_channels=30, # Update this to your actual feature count (vars * levels)
        hidden_channels=64,
        out_channels=30
    )
    model.eval() # Set to evaluation mode (disables gradients)
    
    # 2. Load the graph structure
    graph = torch.load(graph_path, weights_only=False)
    return model, graph

def autoregressive_rollout(model, graph, initial_state, steps=24):
    """
    Generates a forecast by feeding predictions back into the model.
    """
    predictions = []
    current_state = initial_state
    
    with torch.no_grad():
        for step in range(steps):
            # Create the PyG Data object for the current step
            batch = Data(
                x=current_state, 
                edge_index=graph.edge_index, 
                edge_attr=graph.edge_attr
            )
            
            # Predict the next time step
            next_state = model.gnn(batch.x, batch.edge_index)
            predictions.append(next_state.numpy())
            
            # Update the current state for the next loop iteration
            current_state = next_state
            
    return np.stack(predictions)

if __name__ == "__main__":
    # Example Usage: Replace with your actual checkpoint path
    CKPT_PATH = "lightning_logs/version_3/checkpoints/epoch=8-step=6687.ckpt"
    GRAPH_PATH = "data/processed/static_graph.pt"
    
    model, graph = load_model_and_graph(CKPT_PATH, GRAPH_PATH)
    
    # In a real scenario, you would load your normalized zarr dataset here 
    # and extract a specific timestep to serve as initial_state
    # initial_state = load_initial_conditions()
    
    # forecast_array = autoregressive_rollout(model, graph, initial_state, steps=24)
    print("Inference script ready. Forecast generation logic established.")