import os
import numpy as np
import torch
from src.data_processing.downloader import download_era5_data
from src.data_processing.processor import load_and_preprocess
from src.data_processing.to_anemoi import format_for_anemoi
from src.graph_utils.builder import create_spherical_graph

def run_preprocessing_pipeline(year, month):
    raw_path = f"data/raw/era5_{year}_{month}.grib"
    zarr_path = f"data/processed/era5_{year}_{month}.zarr"
    
    # 1. Download (Phase 1)
    if not os.path.exists(raw_path):
        download_era5_data(raw_path, year, month)
    
    # 2. Lazy Load & Clean (Phase 1)
    ds = load_and_preprocess(raw_path)
    
    # 3. Create Graph Metadata (Phase 2)
    # We only need to do this once as the grid is static
    lats = ds.latitude.values
    lons = ds.longitude.values
    # Meshgrid to get every lat/lon pair for the nodes
    lat_grid, lon_grid = np.meshgrid(lats, lons)
    graph_data = create_spherical_graph(lat_grid.flatten(), lon_grid.flatten())
    torch.save(graph_data, "data/processed/static_graph.pt")
    
    # 4. Save to Zarr for Anemoi (Phase 2)
    format_for_anemoi(ds, zarr_path)
    
    return zarr_path, "data/processed/static_graph.pt"