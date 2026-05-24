import xarray as xr
import dask

def load_and_preprocess(file_path):
    # cfgrib engine handles GRIB indexing
    # chunks dictionary defines the dask partition size
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
    
    # Normalize coordinates to ensure consistency for GNN node mapping
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby('longitude')
    
    return ds

def extract_features(ds):
    """
    Example: Calculate Wind Speed from U and V components lazily.
    """
    wind_speed = (ds['u']**2 + ds['v']**2)**0.5
    return wind_speed