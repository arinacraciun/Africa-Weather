import xarray as xr
import numpy as np
import os

def calculate_and_save_global_stats(zarr_path, output_dir):
    print(f"Loading dataset from {zarr_path}...")
    ds = xr.open_zarr(zarr_path)
    
    # 1. Stack into the exact feature layout our model expects
    print("Stacking variables and pressure levels...")
    arr = ds.to_array(dim='variable')
    
    # The resulting dimensions will be: ('features', 'time', 'node')
    arr = arr.stack(features=['variable', 'isobaricInhPa'])
    
    # 2. Compute the mean and std over the 'time' and 'node' dimensions
    # .compute() triggers Dask to actually execute the math across your CPU cores
    print("Calculating global mean (this may take a minute)...")
    global_mean = arr.mean(dim=['time', 'node']).compute().values
    
    print("Calculating global std...")
    global_std = arr.std(dim=['time', 'node']).compute().values
    
    # 3. Save to disk so training and inference scripts can share them
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'train_mean.npy'), global_mean)
    np.save(os.path.join(output_dir, 'train_std.npy'), global_std)
    
    print("Saved train_mean.npy and train_std.npy successfully!")
    print(f"Shape of mean array: {global_mean.shape}") # Should be (30,)

if __name__ == "__main__":
    calculate_and_save_global_stats(
        zarr_path="data/processed/era5_2023_01.zarr",
        output_dir="data/processed/"
    )