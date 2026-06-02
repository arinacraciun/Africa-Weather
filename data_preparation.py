"""
Data Preparation Pipeline

This script handles downloading, preprocessing, formatting, and generating statistics
for both ERA5 atmospheric data and ground truth station data. It outputs data 
in a Zarr format optimized for the Anemoi ML ecosystem, alongside a static 
spherical K-Nearest Neighbors graph for graph neural network training.
"""

import os
from datetime import datetime
import numpy as np

# Data retrieval and processing
import cdsapi
import meteostat as ms
import xarray as xr
import dask

# Graph construction
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

# =============================================================================
# 1. GROUND TRUTH DATA
# =============================================================================

def get_station_data(city_name, lat, lon, start_date, end_date):
    """
    Fetches daily weather observations for the weather station nearest to the given coordinates.
    
    Args:
        city_name (str): Name of the target city.
        lat (float): Latitude of the target location.
        lon (float): Longitude of the target location.
        start_date (datetime): Start date for data retrieval.
        end_date (datetime): End date for data retrieval.
        
    Returns:
        pandas.DataFrame: Daily weather observations for the specified period.
    """
    location = ms.Point(lat, lon)
    nearby_stations = ms.stations.nearby(location, limit=1)
    
    ts = ms.daily(nearby_stations, start_date, end_date)
    data = ts.fetch()
    
    return data

def process_ground_truth_cities(cities_dict, start_date, end_date, output_dir):
    """
    Iterates through a dictionary of cities and saves their station data to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    for city, coords in cities_dict.items():
        df = get_station_data(city, coords[0], coords[1], start_date, end_date)
        output_path = os.path.join(output_dir, f"ground_truth_{city.lower()}.csv")
        df.to_csv(output_path)
        print(f"Saved ground truth for {city} to {output_path}")

# =============================================================================
# 2. ERA5 DATA DOWNLOADER
# =============================================================================

def download_era5_data(output_path, year, month):
    """
    Downloads 3D atmospheric state data from the Copernicus Climate Data Store (CDS).
    
    Args:
        output_path (str): File path to save the raw GRIB data.
        year (str): Target year (e.g., '2023').
        month (str): Target month (e.g., '01').
    """
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': [
                'geopotential', 'temperature', 'specific_humidity',
                'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': [
                '1000', '850', '700', '500', '300', '250'
            ],
            'year': year,
            'month': month,
            'day': [f"{i:02d}" for i in range(1, 32)],
            'time': [f"{i:02d}:00" for i in range(0, 24)],
            'area': [38, -18, -35, 52], # North, West, South, East bounding box for Africa
        },
        output_path
    )
    print(f"ERA5 data downloaded to {output_path}")

# =============================================================================
# 3. DATA PROCESSING & FORMATTING
# =============================================================================

def load_and_preprocess(file_path):
    """
    Lazily loads a GRIB file using Dask and xarray, standardizing the longitude coordinates.
    
    Args:
        file_path (str): Path to the raw GRIB file.
        
    Returns:
        xarray.Dataset: The preprocessed dataset.
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
    
    # Normalize coordinates to ensure consistency for GNN node mapping (-180 to 180)
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby('longitude')
    
    return ds

def extract_features(ds):
    """
    Calculates derived features from the dataset.
    
    Args:
        ds (xarray.Dataset): The preprocessed dataset.
        
    Returns:
        xarray.DataArray: Lazily evaluated wind speed.
    """
    wind_speed = (ds['u']**2 + ds['v']**2)**0.5
    return wind_speed

def format_for_anemoi(ds, output_zarr_path):
    """
    Transforms the dataset into a flattened, time-chunked Zarr format expected 
    by the Anemoi ML ecosystem.
    
    Args:
        ds (xarray.Dataset): The preprocessed dataset.
        output_zarr_path (str): Destination path for the Zarr directory.
        
    Returns:
        xarray.Dataset: The chunked and stacked dataset.
    """
    # Flatten spatial grid into a 1D 'node' dimension and drop invalid geometries
    ds_stacked = ds.stack(node=['latitude', 'longitude']).dropna(dim='node')
    
    # Reset MultiIndex to standard coordinates for Zarr serialization
    ds_stacked = ds_stacked.reset_index('node')
    
    # Rechunk optimally for autoregressive model training
    ds_chunked = ds_stacked.chunk({'time': 1, 'node': -1})
    ds_chunked.to_zarr(output_zarr_path, mode='w')
    
    print(f"Data formatted and saved to Zarr at {output_zarr_path}")
    return ds_chunked

# =============================================================================
# 4. GLOBAL STATISTICS
# =============================================================================

def calculate_and_save_global_stats(zarr_path, output_dir):
    """
    Computes and saves the global mean and standard deviation across the dataset 
    to be used for model normalization during training and inference.
    
    Args:
        zarr_path (str): Path to the processed Zarr dataset.
        output_dir (str): Directory to save the resulting .npy files.
    """
    print(f"Loading dataset from {zarr_path} to compute stats...")
    ds = xr.open_zarr(zarr_path)
    
    # Stack into the exact feature layout our model expects ('features', 'time', 'node')
    arr = ds.to_array(dim='variable')
    arr = arr.stack(features=['variable', 'isobaricInhPa'])
    
    print("Calculating global mean and std (this may take a minute)...")
    global_mean = arr.mean(dim=['time', 'node']).compute().values
    global_std = arr.std(dim=['time', 'node']).compute().values
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'train_mean.npy'), global_mean)
    np.save(os.path.join(output_dir, 'train_std.npy'), global_std)
    
    print(f"Saved global stats. Mean array shape: {global_mean.shape}")

# =============================================================================
# 5. GRAPH CONSTRUCTION
# =============================================================================

def create_spherical_graph(latitudes, longitudes, k_neighbors=6):
    """
    Constructs a K-Nearest Neighbors graph over the surface of a sphere.
    
    Args:
        latitudes (np.ndarray): Array of latitude coordinates.
        longitudes (np.ndarray): Array of longitude coordinates.
        k_neighbors (int): Number of neighbors to connect per node.
        
    Returns:
        torch_geometric.data.Data: Graph structure with edges, attributes, and positions.
    """
    R = 6371.0 # Earth radius in km
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    
    # Convert to 3D Cartesian coordinates
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    
    coords = np.stack([x, y, z], axis=1)
    
    # Calculate connections using Euclidean distance in 3D space
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Establish source and target nodes (slicing from 1 to avoid self-loops)
    src = np.repeat(np.arange(len(coords)), k_neighbors)
    dst = indices[:, 1:].flatten()
    
    edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)
    edge_attr = torch.tensor(distances[:, 1:].flatten(), dtype=torch.float32).unsqueeze(1)
    
    return Data(edge_index=edge_index, edge_attr=edge_attr, pos=torch.tensor(coords, dtype=torch.float32))

# =============================================================================
# 6. MAIN PIPELINE EXECUTION
# =============================================================================

def run_full_pipeline():
    """
    Executes the entire data preparation pipeline:
    1. Fetches ground truth data.
    2. Downloads raw ERA5 data.
    3. Preprocesses and formats ERA5 data to Zarr.
    4. Generates the static spherical graph.
    5. Calculates global dataset statistics.
    """
    # Configuration
    year = '2023'
    months = ['01', '02']
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Ground Truth
    print("--- Phase 1: Ground Truth Extraction ---")
    cities = {
        "Nairobi": (-1.286389, 36.817223),
        "Lagos": (6.465422, 3.406448),
        "Johannesburg": (-26.204103, 28.047305)
    }
    process_ground_truth_cities(
        cities_dict=cities,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        output_dir=processed_dir
    )
    
    # Iterate through target months for ERA5 data
    for month in months:
        print(f"\n--- Phase 2: ERA5 Pipeline for {year}-{month} ---")
        raw_path = os.path.join(raw_dir, f"era5_{year}_{month}.grib")
        zarr_path = os.path.join(processed_dir, f"era5_{year}_{month}.zarr")
        graph_path = os.path.join(processed_dir, "static_graph.pt")
        
        # Download
        if not os.path.exists(raw_path):
            print(f"Downloading raw data to {raw_path}...")
            download_era5_data(raw_path, year, month)
        else:
            print(f"Raw data already exists at {raw_path}, skipping download.")
            
        # Process
        print("Preprocessing raw data...")
        ds = load_and_preprocess(raw_path)
        
        # Graph Construction (Only needed once for static grid)
        if not os.path.exists(graph_path):
            print("Building static spherical graph...")
            lat_grid, lon_grid = np.meshgrid(ds.latitude.values, ds.longitude.values)
            graph_data = create_spherical_graph(lat_grid.flatten(), lon_grid.flatten())
            torch.save(graph_data, graph_path)
            print(f"Graph saved to {graph_path}")
            
        # Format to Anemoi Zarr
        if not os.path.exists(zarr_path):
            print("Formatting dataset for Anemoi...")
            format_for_anemoi(ds, zarr_path)
            
        # Global Stats (Assuming we compute stats based on the first month for normalization)
        if month == months[0]:
            print("\n--- Phase 3: Global Statistics ---")
            calculate_and_save_global_stats(zarr_path, processed_dir)
            
    print("\nData preparation pipeline completed successfully.")

if __name__ == "__main__":
    run_full_pipeline()