import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from src.evaluation.metrics import calculate_deterministic_metrics, create_climatology

def plot_rmse_map(rmse_data, variable_name):
    """Generates a spatial map of the Root Mean Square Error."""
    fig = plt.figure(figsize=(10, 8))
    
    # Use Cartopy to project the data onto a map of Africa
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Plot the RMSE data
    # Selecting the 850 hPa pressure level as an example
    plot_data = rmse_data.sel(isobaricInhPa=850)
    
    mesh = ax.pcolormesh(
        plot_data.longitude, 
        plot_data.latitude, 
        plot_data, 
        cmap='Reds',
        transform=ccrs.PlateCarree()
    )
    
    plt.colorbar(mesh, orientation='horizontal', pad=0.05, label='RMSE')
    plt.title(f'Spatial RMSE for {variable_name} (850 hPa) - Feb 2023')
    
    # Save the figure
    plt.savefig(f'data/reports/rmse_map_{variable_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved RMSE map for {variable_name}")

def plot_acc_timeseries(acc_data, variable_name):
    """Plots how forecast skill degrades over lead time."""
    fig = plt.figure(figsize=(10, 5))
    
    # Selecting the 850 hPa pressure level
    plot_data = acc_data.sel(isobaricInhPa=850)
    
    # X-axis is lead time in hours
    lead_times = range(1, len(plot_data) + 1)
    
    plt.plot(lead_times, plot_data, marker='o', color='blue', linewidth=2)
    plt.axhline(y=0.6, color='red', linestyle='--', label='Useful Skill Threshold (0.6)')
    
    plt.title(f'Anomaly Correlation Coefficient (ACC) for {variable_name} (850 hPa)')
    plt.xlabel('Lead Time (Hours)')
    plt.ylabel('ACC')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(f'data/reports/acc_timeseries_{variable_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved ACC timeseries for {variable_name}")

def print_summary_metrics(forecast_ds, obs_ds, variable_name='t'):
    """
    Calculates single-number summary metrics across the entire spatial grid 
    and forecast duration.
    """
    # Select the specific variable and the 850 hPa pressure level
    fcst = forecast_ds[variable_name].sel(isobaricInhPa=850)
    obs = obs_ds[variable_name].sel(isobaricInhPa=850)
    
    # Calculate global errors
    error = fcst - obs
    
    rmse = float(np.sqrt((error**2).mean().values))
    mae = float(np.abs(error).mean().values)
    bias = float(error.mean().values)
    
    print(f"\n--- Model Performance Summary ({variable_name.upper()} at 850 hPa) ---")
    print(f"Overall RMSE: {rmse:.2f} (Standard deviation of the residuals)")
    print(f"Overall MAE:  {mae:.2f} (Average absolute error magnitude)")
    print(f"Mean Bias:    {bias:.2f} (Positive means model predicts too high, negative means too low)")
    print("------------------------------------------------------\n")


def plot_forecast_vs_actual(forecast_ds, obs_ds, variable_name='t', target_hour=24):
    """
    Plots a 3-panel comparison: Observation, Forecast, and Error for a specific hour.
    """
    # Extract the data for the specific hour and pressure level
    fcst = forecast_ds[variable_name].sel(isobaricInhPa=850).isel(time=target_hour)
    obs = obs_ds[variable_name].sel(isobaricInhPa=850).isel(time=target_hour)
    
    # Calculate the error map
    error = fcst - obs
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f'{variable_name.upper()} Forecast vs Observations at Lead Time +{target_hour} Hours (850 hPa)', fontsize=16)
    
    # Common mapping configurations
    for ax in axes:
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
    # Plot 1: Observation (Ground Truth)
    mesh_obs = axes[0].pcolormesh(obs.longitude, obs.latitude, obs, cmap='coolwarm', transform=ccrs.PlateCarree())
    axes[0].set_title('True Observation (ERA5)')
    plt.colorbar(mesh_obs, ax=axes[0], orientation='horizontal', pad=0.05)
    
    # Plot 2: Model Forecast
    mesh_fcst = axes[1].pcolormesh(fcst.longitude, fcst.latitude, fcst, cmap='coolwarm', transform=ccrs.PlateCarree())
    axes[1].set_title('Model Forecast')
    plt.colorbar(mesh_fcst, ax=axes[1], orientation='horizontal', pad=0.05)
    
    # Plot 3: Error (Forecast - Observation)
    # Using a divergent colormap centered on 0
    max_error = float(np.abs(error).max().values)
    mesh_err = axes[2].pcolormesh(error.longitude, error.latitude, error, cmap='bwr', vmin=-max_error, vmax=max_error, transform=ccrs.PlateCarree())
    axes[2].set_title('Error (Forecast - Truth)')
    plt.colorbar(mesh_err, ax=axes[2], orientation='horizontal', pad=0.05)
    
    plt.tight_layout()
    plt.savefig(f'data/reports/comparison_{variable_name}_hour{target_hour}.png', dpi=300)
    print(f"Saved side-by-side comparison for hour {target_hour}")

def main():
    print("Loading forecast and observation datasets...")
    forecast_ds = xr.open_dataset("data/reports/forecast_feb2023.nc")
    obs_ds = xr.open_dataset("data/reports/observations_feb2023.nc")
    
    # For a true ACC, you need a long-term climatology baseline.
    # Since we only have a short timeframe, we will create a dummy climatology 
    # from the observations just to demonstrate the pipeline execution.
    # In a real scenario, this would be 10+ years of averaged ERA5 data.
    climatology_ds = create_climatology(obs_ds)
    
    print("Calculating spatial metrics...")
    metrics = calculate_deterministic_metrics(forecast_ds, obs_ds, climatology_ds)
    
    print("Generating standard plots...")
    plot_rmse_map(metrics['t']['rmse_map'], 'Temperature')
    plot_acc_timeseries(metrics['t']['acc_timeseries'], 'Temperature')
    
    # --- NEW ADDITIONS ---
    print("\nCalculating summary numerical metrics...")
    print_summary_metrics(forecast_ds, obs_ds, 't')
    
    print("Generating Forecast vs. Actual comparisons...")
    # Plot what the model predicted for Day 1 (hour 24)
    plot_forecast_vs_actual(forecast_ds, obs_ds, 't', target_hour=24)
    # Plot what the model predicted for Day 3 (hour 71)
    plot_forecast_vs_actual(forecast_ds, obs_ds, 't', target_hour=71)
    
    print("All evaluations complete. Check data/reports/ for your final figures.")

if __name__ == "__main__":
    main()