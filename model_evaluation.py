"""
Model Evaluation and Plotting Pipeline

This script handles loading a trained Graph Neural Network, generating an 
autoregressive forecast, reconstructing the 1D graph predictions back into 
2D spatial maps, calculating meteorological metrics (RMSE, ACC), and 
plotting the final visualizations.
"""

import os
import torch
import numpy as np
import xarray as xr
import xskillscore as xs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torch_geometric.data import Data

# Import the model architecture from the previously consolidated training file
from model_training import WeatherLightningModule

# =============================================================================
# 1. DATA PREPARATION & INITIAL STATE
# =============================================================================

def load_and_preprocess(file_path):
    """
    Lazily loads a GRIB file and standardizes longitude coordinates.
    """
    ds = xr.open_dataset(
        file_path, 
        engine='cfgrib', 
        chunks={
            'time': 12, 
            'isobaricInhPa': 1, 
            'latitude': 100, 
            'longitude': 100
        }
    )
    
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby('longitude')
    return ds

def extract_initial_state(obs_ds, start_idx, mean, std):
    """
    Extracts the atmospheric state at a specific time, flattens it for the GNN, 
    and applies the training normalization.
    """
    snap = obs_ds.isel(time=start_idx)
    
    # Stack the 2D grid into the 1D 'node' dimension and drop NaNs
    snap_stacked = snap.stack(node=['latitude', 'longitude']).dropna(dim='node')
    
    # Stack weather variables and pressure levels into 'features'
    arr = snap_stacked.to_array(dim='variable')
    arr = arr.stack(features=['variable', 'isobaricInhPa']).transpose('node', 'features')
    
    # Apply normalization
    val = arr.values
    val_normalized = (val - mean) / (std + 1e-6)
    
    return torch.tensor(val_normalized, dtype=torch.float32)

# =============================================================================
# 2. INFERENCE & RECONSTRUCTION
# =============================================================================

def load_model_and_graph(checkpoint_path, graph_path, num_features=30):
    """
    Loads the trained PyTorch Lightning model and the static PyG graph.
    """
    model = WeatherLightningModule.load_from_checkpoint(
        checkpoint_path,
        in_channels=num_features,
        hidden_channels=64,
        out_channels=num_features
    )
    model.eval() 
    
    graph = torch.load(graph_path, weights_only=False)
    return model, graph

def autoregressive_rollout(model, graph, initial_state, steps=24):
    """
    Generates a forecast by feeding predictions back into the model recursively.
    """
    predictions = []
    current_state = initial_state
    
    with torch.no_grad():
        for _ in range(steps):
            batch = Data(
                x=current_state, 
                edge_index=graph.edge_index, 
                edge_attr=graph.edge_attr
            )
            
            next_state = model.gnn(batch.x, batch.edge_index)
            predictions.append(next_state.numpy())
            current_state = next_state
            
    return np.stack(predictions)

def reconstruct_forecast(predictions, template_ds, mean, std):
    """
    Converts 1D graph predictions back into a multi-dimensional xarray Dataset.
    """
    unnormalized_preds = (predictions * std) + mean
    
    lats = template_ds.latitude.values
    lons = template_ds.longitude.values
    times = template_ds.time.values[:len(predictions)]
    var_names = list(template_ds.data_vars.keys())
    
    num_lats = len(lats)
    num_lons = len(lons)
    num_vars = len(var_names)
    num_levels = len(template_ds.isobaricInhPa)
    
    # Reshape back to 5D: (time, lat, lon, variable, level)
    reshaped = unnormalized_preds.reshape(
        len(times), num_lats, num_lons, num_vars, num_levels
    )
    
    # Transpose to standard format: (time, variable, level, lat, lon)
    reshaped = np.transpose(reshaped, (0, 3, 4, 1, 2))
    
    ds_forecast = xr.DataArray(
        reshaped,
        coords={
            'time': times,
            'variable': var_names, 
            'isobaricInhPa': template_ds.isobaricInhPa.values,
            'latitude': lats,
            'longitude': lons
        },
        dims=['time', 'variable', 'isobaricInhPa', 'latitude', 'longitude']
    ).to_dataset(dim='variable')
    
    return ds_forecast

# =============================================================================
# 3. METRICS CALCULATION
# =============================================================================

def create_climatology(historical_ds):
    """
    Creates a simple baseline by averaging historical data by day of the year.
    """
    return historical_ds.groupby('time.dayofyear').mean('time')

def calculate_deterministic_metrics(forecast_ds, obs_ds, climatology_ds):
    """
    Evaluates the forecast against true observations and a climatological baseline.
    """
    metrics = {}
    forecast_ds, obs_ds = xr.align(forecast_ds, obs_ds, join='inner')
    
    for var in forecast_ds.data_vars:
        fcst = forecast_ds[var]
        obs = obs_ds[var]
        clim = climatology_ds[var]
        
        # Spatial RMSE over time
        rmse_map = xs.rmse(fcst, obs, dim='time')
        
        # Anomaly Correlation Coefficient (ACC)
        fcst_anomaly = fcst - clim
        obs_anomaly = obs - clim
        acc_timeseries = xs.pearson_r(fcst_anomaly, obs_anomaly, dim=['latitude', 'longitude'])
        
        metrics[var] = {
            'rmse_map': rmse_map,
            'acc_timeseries': acc_timeseries
        }
        
    return metrics

def print_summary_metrics(forecast_ds, obs_ds, variable_name='t', level=850):
    """
    Calculates single-number summary metrics across the entire spatial grid.
    """
    fcst = forecast_ds[variable_name].sel(isobaricInhPa=level)
    obs = obs_ds[variable_name].sel(isobaricInhPa=level)
    
    error = fcst - obs
    rmse = float(np.sqrt((error**2).mean().values))
    mae = float(np.abs(error).mean().values)
    bias = float(error.mean().values)
    
    print(f"\n--- Model Performance Summary ({variable_name.upper()} at {level} hPa) ---")
    print(f"Overall RMSE: {rmse:.2f}")
    print(f"Overall MAE:  {mae:.2f}")
    print(f"Mean Bias:    {bias:.2f}")
    print("------------------------------------------------------\n")

# =============================================================================
# 4. PLOTTING CAPABILITIES
# =============================================================================

def plot_rmse_map(rmse_data, variable_name, level=850, output_dir="data/reports"):
    """
    Generates a Cartopy spatial map of the Root Mean Square Error.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    plot_data = rmse_data.sel(isobaricInhPa=level)
    mesh = ax.pcolormesh(
        plot_data.longitude, 
        plot_data.latitude, 
        plot_data, 
        cmap='Reds',
        transform=ccrs.PlateCarree()
    )
    
    plt.colorbar(mesh, orientation='horizontal', pad=0.05, label='RMSE')
    plt.title(f'Spatial RMSE for {variable_name} ({level} hPa) - Feb 2023')
    plt.savefig(os.path.join(output_dir, f'rmse_map_{variable_name}.png'), dpi=300, bbox_inches='tight')
    print(f"Saved RMSE map for {variable_name}")

def plot_acc_timeseries(acc_data, variable_name, level=850, output_dir="data/reports"):
    """
    Plots forecast skill degradation over lead time.
    """
    fig = plt.figure(figsize=(10, 5))
    plot_data = acc_data.sel(isobaricInhPa=level)
    lead_times = range(1, len(plot_data) + 1)
    
    plt.plot(lead_times, plot_data, marker='o', color='blue', linewidth=2)
    plt.axhline(y=0.6, color='red', linestyle='--', label='Useful Skill Threshold (0.6)')
    
    plt.title(f'Anomaly Correlation Coefficient (ACC) for {variable_name} ({level} hPa)')
    plt.xlabel('Lead Time (Hours)')
    plt.ylabel('ACC')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f'acc_timeseries_{variable_name}.png'), dpi=300, bbox_inches='tight')
    print(f"Saved ACC timeseries for {variable_name}")

def plot_forecast_vs_actual(forecast_ds, obs_ds, variable_name='t', target_hour=24, level=850, output_dir="data/reports"):
    """
    Plots a 3-panel comparison: Observation, Forecast, and Error.
    """
    fcst = forecast_ds[variable_name].sel(isobaricInhPa=level).isel(time=target_hour)
    obs = obs_ds[variable_name].sel(isobaricInhPa=level).isel(time=target_hour)
    error = fcst - obs
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f'{variable_name.upper()} Forecast vs Observations at Lead Time +{target_hour} Hours ({level} hPa)', fontsize=16)
    
    for ax in axes:
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
    # Plot 1: Observation
    mesh_obs = axes[0].pcolormesh(obs.longitude, obs.latitude, obs, cmap='coolwarm', transform=ccrs.PlateCarree())
    axes[0].set_title('True Observation (ERA5)')
    plt.colorbar(mesh_obs, ax=axes[0], orientation='horizontal', pad=0.05)
    
    # Plot 2: Forecast
    mesh_fcst = axes[1].pcolormesh(fcst.longitude, fcst.latitude, fcst, cmap='coolwarm', transform=ccrs.PlateCarree())
    axes[1].set_title('Model Forecast')
    plt.colorbar(mesh_fcst, ax=axes[1], orientation='horizontal', pad=0.05)
    
    # Plot 3: Error
    max_error = float(np.abs(error).max().values)
    mesh_err = axes[2].pcolormesh(error.longitude, error.latitude, error, cmap='bwr', vmin=-max_error, vmax=max_error, transform=ccrs.PlateCarree())
    axes[2].set_title('Error (Forecast - Truth)')
    plt.colorbar(mesh_err, ax=axes[2], orientation='horizontal', pad=0.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{variable_name}_hour{target_hour}.png'), dpi=300)
    print(f"Saved side-by-side comparison for hour {target_hour}")

# =============================================================================
# 5. MAIN EVALUATION PIPELINE
# =============================================================================

def run_evaluation_pipeline():
    """
    Orchestrates the entire inference, evaluation, and plotting workflow.
    """
    # Configuration
    CKPT_PATH = "lightning_logs/version_6/checkpoints/epoch=12-step=9659.ckpt" 
    GRAPH_PATH = "data/processed/static_graph.pt"
    TEST_DATA_PATH = "data/raw/era5_africa_2023_02.grib"
    REPORTS_DIR = "data/reports"
    FORECAST_HOURS = 72 
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    TRAIN_MEAN = np.load("data/processed/train_mean.npy")
    TRAIN_STD = np.load("data/processed/train_std.npy")
    
    # 1. Load Model and Raw Test Observations
    print("Loading model and observations...")
    model, graph = load_model_and_graph(CKPT_PATH, GRAPH_PATH, num_features=30)
    obs_ds_full = load_and_preprocess(TEST_DATA_PATH)
    
    # 2. Extract Initial State (Hour 0)
    initial_state = extract_initial_state(obs_ds_full, start_idx=0, mean=TRAIN_MEAN, std=TRAIN_STD)
    
    # 3. Generate Forecast
    print(f"Running autoregressive forecast for {FORECAST_HOURS} hours...")
    raw_predictions = autoregressive_rollout(model, graph, initial_state, steps=FORECAST_HOURS)
    
    # 4. Reconstruct Forecast to Xarray 
    print("Reconstructing predictions into 2D maps...")
    template_ds = obs_ds_full.isel(time=slice(1, FORECAST_HOURS + 1)) 
    forecast_ds = reconstruct_forecast(raw_predictions, template_ds, mean=TRAIN_MEAN, std=TRAIN_STD)
    
    # Align observation timeframe with forecast timeframe
    obs_ds = obs_ds_full.isel(time=slice(1, FORECAST_HOURS + 1))
    
    # Save netCDF artifacts
    forecast_ds.to_netcdf(os.path.join(REPORTS_DIR, "forecast_feb2023.nc"))
    obs_ds.to_netcdf(os.path.join(REPORTS_DIR, "observations_feb2023.nc"))
    
    # 5. Calculate Metrics
    print("\nCalculating metrics...")
    climatology_ds = create_climatology(obs_ds)
    metrics = calculate_deterministic_metrics(forecast_ds, obs_ds, climatology_ds)
    
    print_summary_metrics(forecast_ds, obs_ds, variable_name='t')
    
    # 6. Generate Plots
    print("Generating visualizations...")
    plot_rmse_map(metrics['t']['rmse_map'], 'Temperature', output_dir=REPORTS_DIR)
    plot_acc_timeseries(metrics['t']['acc_timeseries'], 'Temperature', output_dir=REPORTS_DIR)
    
    plot_forecast_vs_actual(forecast_ds, obs_ds, 't', target_hour=24, output_dir=REPORTS_DIR)
    plot_forecast_vs_actual(forecast_ds, obs_ds, 't', target_hour=71, output_dir=REPORTS_DIR)
    
    print(f"\nAll evaluations complete. Artifacts saved in {REPORTS_DIR}/")

if __name__ == "__main__":
    run_evaluation_pipeline()