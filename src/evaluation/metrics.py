import xarray as xr
import xskillscore as xs

def calculate_deterministic_metrics(forecast_ds, obs_ds, climatology_ds):
    """
    Evaluates the forecast against true observations and a climatological baseline.
    """
    metrics = {}
    
    # Ensure datasets align in time and space
    forecast_ds, obs_ds = xr.align(forecast_ds, obs_ds, join='inner')
    
    for var in forecast_ds.data_vars:
        fcst = forecast_ds[var]
        obs = obs_ds[var]
        clim = climatology_ds[var]
        
        # 1. Spatial RMSE over time
        # This returns a 2D map showing regions of high error
        rmse_map = xs.rmse(fcst, obs, dim='time')
        
        # 2. Anomaly Correlation Coefficient (ACC)
        # Calculate anomalies by subtracting the long-term climatology mean
        fcst_anomaly = fcst - clim
        obs_anomaly = obs - clim
        
        # Calculate Pearson correlation over the spatial dimensions
        # This returns a 1D array over time, showing how skill degrades as forecast lead time increases
        acc_timeseries = xs.pearson_r(fcst_anomaly, obs_anomaly, dim=['latitude', 'longitude'])
        
        metrics[var] = {
            'rmse_map': rmse_map,
            'acc_timeseries': acc_timeseries
        }
        
    return metrics

def create_climatology(historical_ds):
    """
    Creates a simple baseline by averaging historical data by day of the year.
    """
    return historical_ds.groupby('time.dayofyear').mean('time')