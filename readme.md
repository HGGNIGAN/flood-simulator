# Flood Simulator

A Python-based flood simulation pipeline that generates synthetic storm data and simulates flood inundation for machine learning applications.

## Prerequisites

### System Requirements

- Python 3.8 or higher
- 32GB RAM (recommended for optimal performance)
- (WIP) Docker (optional, required for LISFLOOD hydraulic simulation)

### Python Libraries

```bash
pip install numpy
pip install rasterio
pip install pandas
pip install noise
pip install noise
```

**Details:**

- **numpy**: Numerical computing library for array operations and mathematical functions
- **rasterio**: Geospatial raster data I/O library for reading/writing GeoTIFF files
- **pandas**: Data manipulation and analysis library for CSV operations
- **noise**: Perlin noise generation library for synthetic storm patterns

### (WIP) Docker Setup (Optional - For LISFLOOD Simulation)

If you want to use the actual LISFLOOD hydraulic simulation instead of the demo simulation:

1. **Install Docker:**
   - Follow the official installation guide: <https://docs.docker.com/get-docker/>
   - Verify installation: `docker --version`

2. **Pull LISFLOOD Docker Image:**

   ```python
   docker pull jrce1/lisflood:latest
   ```

3. **Enable Docker Mode:**

   - In `src/main.py`, set `USE_DOCKER = True`
   - Note: LISFLOOD requires extensive configuration (DEM, roughness maps, PCRaster format support, etc.)

**Note:** The default demo simulation mode (`USE_DOCKER = False`) does not require Docker and provides a simplified hydrological model suitable for development and testing.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/HGGNIGAN/flood-simulator.git
   cd flood-simulator
   ```

2. Install required Python libraries:

   ```python
   pip install numpy rasterio pandas noise
   ```

3. (Optional) Set up Docker if using LISFLOOD simulation **(Still work in progress, not currently available)**

## Project Structure

```text
flood-simulator/
├── data/
│   ├── static/              # Static input data (DEM, roughness, etc.)
│   ├── storm_generator/     # Generated storm precipitation data
│   ├── simulation_output/   # Flood simulation results
│   ├── lisflood_settings/   # LISFLOOD configuration files
│   ├── feature_data.csv     # Extracted feature data
│   └── flood_timeseries.csv # Flood time series for ML training
├── src/
│   ├── main.py                    # Main pipeline orchestrator
│   ├── flood_simulator/           # Flood simulation module
│   ├── storm_gen/                 # Storm generation module
│   ├── lisflood_docker/           # LISFLOOD Docker integration
│   └── rasterizer/                # Data extraction modules
└── readme.md
```

## Usage

### 1. Generate Storm Data

```bash
python -m src.storm_gen.storm_generator
```

### 2. Run Flood Simulation Pipeline

```bash
python -m src.main
```

### 3. Extract Feature Data

```bash
python -m src.rasterizer.data_rasterizer
```

### 4. Extract Flood Time Series

```bash
python -m src.rasterizer.data_target_rasterizer
```

## License

TODO

## Contributing

TODO
