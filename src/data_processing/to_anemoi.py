import xarray as xr

def format_for_anemoi(ds, output_zarr_path):
    """
    Transforms the lazy xarray dataset into the flattened, time-chunked 
    Zarr format expected by the Anemoi ML ecosystem.
    """
    # 1. Flatten the spatial grid into a 1D 'node' dimension
    ds_stacked = ds.stack(node=['latitude', 'longitude'])
    
    # 2. Drop invalid geometries (e.g., if the bounding box has NaN edges)
    ds_stacked = ds_stacked.dropna(dim='node')
    
    # 3. FIX: Reset the MultiIndex to standard coordinates so Zarr can serialize it
    ds_stacked = ds_stacked.reset_index('node')
    
    # 4. Rechunk optimally for autoregressive model training
    # Loading time=1 gets the full spatial extent of the nodes at once
    ds_chunked = ds_stacked.chunk({'time': 1, 'node': -1})
    
    # 5. Export to Zarr
    ds_chunked.to_zarr(output_zarr_path, mode='w')
    
    return ds_chunked