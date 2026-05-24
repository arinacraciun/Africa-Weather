import xarray as xr
import torch
import numpy as np
import os
from src.data_processing.processor import load_and_preprocess
from src.evaluation.inference import load_model_and_graph, autoregressive_rollout
from src.evaluation.reconstruct import reconstruct_forecast
from src.evaluation.metrics import calculate_deterministic_metrics

def prepare_test_data(raw_grib_path):
    """Loads the test data and prepares it as obs_ds."""
    # Load and clean the GRIB file (same as Phase 1)
    ds = load_and_preprocess(raw_grib_path)
    return ds

def extract_initial_state(obs_ds, start_idx=0, mean=0, std=1):
    """
    Extracts the atmospheric state at t=0, flattens it for the GNN, 
    and applies the exact same normalization used during training.
    """
    snap = obs_ds.isel(time=start_idx)
    
    # 1. FIX: Stack the 2D grid into the 1D 'node' dimension first
    snap_stacked = snap.stack(node=['latitude', 'longitude'])
    
    # Drop any NaN nodes (matches the Phase 2 Zarr processing logic)
    snap_stacked = snap_stacked.dropna(dim='node')
    
    # 2. Convert to array and stack the weather variables/levels into 'features'
    arr = snap_stacked.to_array(dim='variable')
    arr = arr.stack(features=['variable', 'isobaricInhPa']).transpose('node', 'features')
    
    # 3. Apply the exact same mean and std from your training dataset
    val = arr.values
    val_normalized = (val - mean) / (std + 1e-6)
    
    return torch.tensor(val_normalized, dtype=torch.float32)

def main():
    # --- 1. Configuration ---
    CKPT_PATH = "lightning_logs/version_4/checkpoints/epoch=14-step=11145.ckpt" 
    GRAPH_PATH = "data/processed/static_graph.pt"
    TEST_DATA_PATH = "data/raw/era5_africa_2023_02.grib"
    FORECAST_HOURS = 72 # 3 days of hourly forecasts
    
    # In run_evaluation.py
    TRAIN_MEAN = np.load("data/processed/train_mean.npy")
    TRAIN_STD = np.load("data/processed/train_std.npy")
    
    # --- 2. Load the World ---
    print("Loading model and observations...")
    model, graph = load_model_and_graph(CKPT_PATH, GRAPH_PATH)
    obs_ds_full = prepare_test_data(TEST_DATA_PATH)
    
    # --- 3. Prepare Initial State ---
    # We use Feb 1, 00:00 as our starting point
    initial_state = extract_initial_state(obs_ds_full, start_idx=0, mean=TRAIN_MEAN, std=TRAIN_STD)
    
    # --- 4. Run the Forecast ---
    print(f"Running autoregressive forecast for {FORECAST_HOURS} hours...")
    raw_predictions = autoregressive_rollout(model, graph, initial_state, steps=FORECAST_HOURS)
    
    # --- 5. Reconstruct forecast_ds ---
    print("Reconstructing predictions into 2D maps...")
    # Pass the first 'FORECAST_HOURS' of obs_ds as the template to get the right coordinates
    template_ds = obs_ds_full.isel(time=slice(1, FORECAST_HOURS + 1)) 
    
    forecast_ds = reconstruct_forecast(
        raw_predictions, 
        template_ds, 
        mean=TRAIN_MEAN, 
        std=TRAIN_STD
    )
    
    # --- 6. Finalize obs_ds ---
    # Our predictions are for hour 1 through 72. 
    # We must slice the observations to match exactly so xskillscore can compare them.
    obs_ds = obs_ds_full.isel(time=slice(1, FORECAST_HOURS + 1))
    
    print("forecast_ds and obs_ds successfully generated!")
    print(f"Forecast shape: {forecast_ds.t.shape}")
    print(f"Observation shape: {obs_ds.t.shape}")

    # Ensure the directory exists before saving
    os.makedirs("data/reports", exist_ok=True)
    
    # Save them to disk so you don't have to re-run inference just to plot charts
    forecast_ds.to_netcdf("data/reports/forecast_feb2023.nc")
    obs_ds.to_netcdf("data/reports/observations_feb2023.nc")

if __name__ == "__main__":
    main()