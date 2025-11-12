# Flood Simulator

A Python-based flood simulation pipeline that generates realistic synthetic storm data and simulates flood inundation using physics-based hydraulics for machine learning applications.

## Features

- **Realistic Storm Generation**: 3D Perlin noise-based precipitation with storm movement, wind effects, and vorticity
- **Physics-Based Simulation**: Manning's equation and infiltration modeling for accurate flood depth prediction
- **High Performance**: Numba JIT compilation with parallel processing for fast computation
- **Multiple Time Windows**: Generates current, 1h, 2h, and 24h cumulative precipitation
- **Optimized Data Processing**: Vectorized rasterization with parallel frame processing
- **ML-Ready Outputs**: CSV time series extraction for training flood prediction models

## Prerequisites

### System Requirements

- Python 3.8 or higher
- 8GB RAM minimum, 16GB+ recommended for large datasets
- Multi-core CPU recommended for parallel processing (auto-detects available cores)

### Python Libraries

```bash
pip install numpy rasterio polars noise tqdm numba
```

**Details:**

- **numpy**: Numerical computing and array operations
- **rasterio**: Geospatial raster I/O for GeoTIFF files
- **polars**: High-performance DataFrame library (faster than pandas)
- **noise**: Perlin noise generation for synthetic storm patterns
- **tqdm**: Progress bars for long-running operations
- **numba**: JIT compilation for 10-100x speedup (optional but highly recommended)

### Optional Dependencies

- **Docker** (WIP): For running LISFLOOD hydraulic simulation (not required for default physics-based simulator)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/HGGNIGAN/flood-simulator.git
   cd flood-simulator
   ```

2. Install required Python libraries:

   ```bash
   pip install numpy rasterio polars noise tqdm numba
   ```

## Project Structure

```text
flood-simulator/
├── data/
│   ├── static/                    # Static input data (DEM, LULC, etc.)
│   ├── storm_generator/           # Generated precipitation data
│   │   ├── current_precip/        # Instantaneous precipitation (mm/hr)
│   │   ├── total_1h/              # 1-hour cumulative precipitation (mm)
│   │   ├── total_2h/              # 2-hour cumulative precipitation (mm)
│   │   └── total_24h/             # 24-hour cumulative precipitation (mm)
│   ├── simulation_output/         # Flood simulation results
│   │   ├── flood_depth_raw_*.tif  # Raw flood depth outputs (m)
│   │   └── flood_classified_*.tif # Classified flood intensity (0-3)
│   ├── lisflood_settings/         # LISFLOOD configuration files (optional)
│   ├── precipitation_*.csv        # Extracted precipitation time series
│   └── target_value.csv           # Flood classification time series for ML
├── src/
│   ├── main.py                    # Main pipeline orchestrator
│   ├── flood_simulator/
│   │   └── simulator.py           # Physics-based flood simulator (Manning's equation)
│   ├── storm_gen/
│   │   └── storm_generator.py     # Perlin noise-based storm generator
│   ├── lisflood_docker/           # LISFLOOD Docker integration (WIP)
│   │   ├── runner.py              # Docker container management
│   │   └── config_gen.py          # LISFLOOD configuration generator
│   └── rasterizer/
│       ├── data_rasterizer.py     # Feature extraction (DEM, elevation, etc.)
│       ├── data_storm_rasterizer.py   # Precipitation time series extraction
│       └── data_target_rasterizer.py  # Flood classification extraction (optimized)
└── readme.md
```

## Usage

### Quick Start: Full Pipeline

Run the complete simulation pipeline (storm generation + flood simulation + classification):

```bash
python src/main.py
```

This will:

1. Generate realistic storm data with movement and wind effects
2. Simulate flood depths using physics-based hydraulics
3. Classify flood intensity (0=none, 1=light, 2=moderate, 3=severe)
4. Output results to `data/simulation_output/`

### Individual Components

#### 1. Generate Storm Data Only

```bash
python src/storm_gen/storm_generator.py
```

**Configuration** (edit in `storm_generator.py`):

- `SIMULATION_HOURS = 8` - Total simulation duration
- `TIME_STEP_MINUTES = 15` - Time between frames
- `MAX_INTENSITY = 300.0` - Maximum precipitation (mm/hr)
- `STORM_DIRECTION = 50` - Storm movement direction (degrees)
- `STORM_SPEED = 0.3` - Movement speed (pixels/frame)
- `ENABLE_VORTICITY = True` - Enable rotation/cyclonic effects

**Outputs**: 4 directories with time-series GeoTIFF files:

- `data/storm_generator/current_precip/` - Instantaneous rates
- `data/storm_generator/total_1h/` - 1-hour totals
- `data/storm_generator/total_2h/` - 2-hour totals  
- `data/storm_generator/total_24h/` - 24-hour totals

#### 2. Extract Precipitation Time Series

```bash
python src/rasterizer/data_storm_rasterizer.py
```

**Outputs**: CSV files with precipitation values at each pixel location over time

- `data/precipitation_current.csv`
- `data/precipitation_1h.csv`
- `data/precipitation_2h.csv`
- `data/precipitation_24h.csv`

#### 3. Extract Spatial Features

```bash
python src/rasterizer/data_rasterizer.py
```

**Outputs**: `data/feature_data.csv` with spatial features:

- Coordinates (x, y)
- Elevation (from DEM)
- Land use classification
- Other static spatial features

#### 4. Extract Flood Time Series

```bash
python src/rasterizer/data_target_rasterizer.py
```

**Configuration**:

- `USE_PARALLEL = True` - Enable parallel processing (recommended)
- `USE_VECTORIZATION = True` - Enable vectorized extraction (recommended)
- `NUM_WORKERS = None` - CPU cores (None = auto-detect)

**Outputs**: `data/flood_timeseries.csv` with flood classification at each location over time

## Performance Optimization

### Storm Generator

- **Parallel Processing**: Uses chunked processing optimized for Windows
- **Vectorization**: NumPy meshgrid operations for coordinate transforms
- **Typical Runtime**: 2-5 minutes for 96 frames (500x500 pixels)

### Flood Simulator  

- **Numba JIT**: Compiles critical functions to machine code (10-100x speedup)
- **Parallel Execution**: Multi-threaded flow calculations
- **Typical Runtime**: 30-90 seconds per frame

### Data Extraction

- **Vectorized Processing**: NumPy array indexing instead of loops (10-50x faster)
- **Parallel Frame Processing**: Multiple CPU cores extract frames simultaneously
- **Typical Runtime**: 30-90 seconds for 96 frames, 100k pixels (with parallel mode)

## Configuration

### Storm Parameters

Key parameters in `src/storm_gen/storm_generator.py`:

```python
MAX_INTENSITY = 300.0           # Maximum precipitation (mm/hr)
INTENSITY_THRESHOLD = 0.35      # Creates distinct storm cells
INTENSITY_CONCENTRATION = 2.0   # Storm core intensity (higher = more concentrated)
STORM_PEAK_POSITION = 0.4       # Peak at 40% through simulation
STORM_DIRECTION = 50            # Movement direction (degrees)
STORM_SPEED = 0.3               # Movement speed (pixels/frame)
WIND_STRENGTH = 0.5             # Wind influence (0-1)
VORTICITY_STRENGTH = 0.1        # Rotation strength (0-0.2)
```

### Simulator Parameters

Key parameters in `src/flood_simulator/simulator.py`:

```python
TIME_STEP_SECONDS = 10.0        # Computational time step
SIMULATION_DURATION_SECONDS = 7200.0  # Total time (2 hours)
CELL_SIZE = 30.0                # Grid resolution (meters)
MIN_DEPTH_THRESHOLD = 0.001     # Minimum depth to consider (m)
```

## Troubleshooting

### Storm Generator Issues

- **Low maximum values**: Check `INTENSITY_CONCENTRATION` (should be 2.0, not higher)
- **No storm movement**: Verify `STORM_SPEED > 0` and `STORM_DIRECTION` is set
- **Slow performance**: Enable `USE_PARALLEL = True` and increase `CHUNK_SIZE`

### Simulator Issues

- **Underestimated flood depths**: Check infiltration parameters and elevation/slope reduction factors
- **Out of memory**: Reduce grid size or use lower resolution DEM
- **Numba import error**: Install with `pip install numba` (optional but highly recommended)

### Data Extraction Issues

- **Slow extraction**: Enable `USE_PARALLEL = True` and `USE_VECTORIZATION = True`
- **Memory errors on Windows**: Parallel processing uses chunked approach to avoid WinError 1450
- **Wrong file pattern**: Check glob patterns match actual filenames (e.g., `*_frame_*.tif`)

## Technical Details

### Physics Models

**Manning's Equation** (Surface Flow):

```
Q = (1/n) × A × R_h^(2/3) × S^(1/2)
```

- Q = flow rate (m³/s)
- n = Manning's roughness coefficient
- A = flow area (m²)
- R_h = hydraulic radius (m)
- S = water surface slope

**Green-Ampt Infiltration** (Simplified):

```
f = K_sat × (1 - imperviousness)
```

- f = infiltration rate (m/s)
- K_sat = saturated hydraulic conductivity
- imperviousness = fraction of impervious surface (0-1)

### Storm Generation

Uses 3D Perlin noise with:

- **Spatial dimensions**: X, Y coordinates
- **Temporal dimension**: Frame number × TIME_SCALE
- **Movement**: Offset coordinates by direction vector × speed
- **Wind effects**: Additional distortion along movement direction
- **Vorticity**: Rotational transform around storm center

## License

TODO

## Contributing

TODO
