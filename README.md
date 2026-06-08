
## Physics-Informed Graph Neural Networks for Atmospheric Forecasting

This repository contains a complete, end-to-end machine learning pipeline for processing massive meteorological datasets, training a Graph Neural Network (GNN) to simulate atmospheric dynamics, and evaluating autoregressive weather forecasts. The system is designed to predict multi-variable weather states over the African continent using ERA5 reanalysis data and ground-truth station observations.

### **Project Overview**

Modern data-driven weather models must overcome significant challenges: planetary-scale data volume, polar distortion on standard grids, and exponential error accumulation during multi-step forecasts. This project addresses these via an out-of-core data pipeline, a spherical graph topology, and a physics-informed residual modeling approach.

### **Pipeline Architecture**

| Component | File | Responsibilities | Core Technologies |
| --- | --- | --- | --- |
| **Data Engineering** | `data_preparation.py` | ERA5 downloading, lazy loading, Zarr chunking, and spherical graph generation. | `cdsapi`, `xarray`, `dask`, `scikit-learn` |
| **Model Training** | `model_training.py` | PyG data collation, physics-informed loss computation, and mixed-precision training. | `pytorch`, `pytorch_lightning`, `torch_geometric` |
| **Evaluation** | `model_evaluation.py` | Autoregressive inference, graph-to-grid reconstruction, metric calculation, and visualization. | `xskillscore`, `cartopy`, `matplotlib` |

---

### **Core Technical Achievements**

* **Massive Data Handling (Out-of-Core Processing):** Implemented lazy loading and optimized chunking using `xarray` and `dask` to process high-resolution, multi-dimensional ERA5 GRIB files without exhausting system memory. Data is serialized into a flattened, Anemoi-compatible Zarr format for high-throughput model ingestion.
* **Spherical Graph Construction:** Solved the "pole problem" inherent in 2D latitude/longitude grids by mapping coordinates to 3D Cartesian space. Built a K-Nearest Neighbors (KNN) mesh using true spatial Euclidean distances to ensure accurate message passing globally.
* **Physics-Informed Architecture:** Engineered a GNN that predicts atmospheric *residuals* (the delta between states) rather than absolute values, preventing numerical explosion during deep autoregressive rollouts. Integrated a custom Graph Dirichlet smoothness penalty to suppress physically impossible gradient spikes and checkerboard artifacts.
* **Rigorous Meteorological Evaluation:** Built a strict evaluation protocol that reconstructs 1D graph outputs back into isolated 5D spatial tensors to prevent multi-variable data bleed. Benchmarks model skill using domain-standard metrics like Spatial Root Mean Square Error (RMSE) and Anomaly Correlation Coefficient (ACC) against a computed climatological baseline.
* **Hardware Optimization:** Utilized Brain Floating Point 16 (`bf16-mixed`) precision via PyTorch Lightning to aggressively reduce memory footprint and latency while avoiding underflow instability during gradient calculations.

---

### **Future Roadmap**

This pipeline serves as the foundation for an advanced forecasting system. Planned architectural upgrades include:

* **Temporal Context Injection:** Integrating cyclical embeddings (sine/cosine waves for hour-of-day and day-of-year) to allow the model to learn diurnal and seasonal physical cycles.
* **Encoder-Processor-Decoder Migration:** Expanding the model's receptive field by transitioning from a local KNN mesh to a multi-scale hierarchical graph, allowing macro-scale storm systems to propagate globally in a single forward pass.
* **Probabilistic Forecasting:** Shifting the objective function to Gaussian Negative Log-Likelihood (NLL) to predict both mean and variance, enabling ensemble generation and uncertainty quantification.
* **Local Downscaling & Bias Correction:** Implementing a routing mechanism to map specific graph node predictions to an auxiliary Multi-Layer Perceptron (MLP) trained on `meteostat` station data, correcting local terrain biases in target cities like Nairobi, Lagos, and Johannesburg.

---

### **Usage**

**1. Install dependencies with Mamba/Conda Environment**

For a robust environment setup that automatically handles complex geospatial C++ binaries (such as `eccodes` for GRIB files and `proj` for Cartopy maps), you can use the provided `environment.yml` file with **Mamba** or **Conda**.

This configuration prioritizes `conda-forge` to cleanly resolve heavy deep learning and meteorological dependencies, and bridges `meteostat` safely via `pip`.

Run the following commands in your terminal to build and activate the environment:

```bash
# Create the environment from the yaml specification
mamba env create -f environment.yml

# Activate the environment
mamba activate weather_gnn

```

*(Note: If you do not have Mamba installed, you can replace `mamba` with `conda` in the commands above.)*

**2. Data Preparation**
Run the data pipeline to download ERA5 data, generate the static graph, and compile training statistics.
`python data_preparation.py`

**3. Model Training**
Execute the PyTorch Lightning training loop.
`python model_training.py`

**4. Evaluation & Visualization**
Generate an autoregressive rollout and output Cartopy spatial maps and ACC timeseries.
`python model_evaluation.py`