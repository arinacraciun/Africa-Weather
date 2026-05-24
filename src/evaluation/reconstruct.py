import xarray as xr
import numpy as np

def reconstruct_forecast(predictions, template_ds, mean, std):
    """
    Converts 1D graph predictions back into a multi-dimensional xarray Dataset.
    
    predictions: numpy array of shape (time, nodes, features)
    template_ds: The original xarray dataset before flattening (used for coordinates)
    mean, std: The normalization arrays used in the Datamodule
    """
    # 1. Un-normalize the data
    unnormalized_preds = (predictions * std) + mean
    
    # 2. Extract spatial dimensions from the template
    lats = template_ds.latitude.values
    lons = template_ds.longitude.values
    times = template_ds.time.values[:len(predictions)]
    
    # FIX: Get the actual variable names from the dataset's data_vars
    var_names = list(template_ds.data_vars.keys())
    
    num_lats = len(lats)
    num_lons = len(lons)
    num_vars = len(var_names)
    num_levels = len(template_ds.isobaricInhPa)
    
    # 3. Reshape the flat node/feature arrays back to 5D
    # Shape becomes: (time, lat, lon, variable, level)
    reshaped = unnormalized_preds.reshape(
        len(times), num_lats, num_lons, num_vars, num_levels
    )
    
    # 4. Transpose to match standard meteorological formats: 
    # (time, variable, isobaricInhPa, latitude, longitude)
    reshaped = np.transpose(reshaped, (0, 3, 4, 1, 2))
    
    # 5. Build the new xarray Dataset
    ds_forecast = xr.DataArray(
        reshaped,
        coords={
            'time': times,
            'variable': var_names, # Use the extracted variable names here
            'isobaricInhPa': template_ds.isobaricInhPa.values,
            'latitude': lats,
            'longitude': lons
        },
        dims=['time', 'variable', 'isobaricInhPa', 'latitude', 'longitude']
    ).to_dataset(dim='variable')
    
    return ds_forecast