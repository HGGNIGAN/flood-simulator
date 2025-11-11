"""
Storm Generator Script (Perlin Noise Based)
Generates synthetic precipitation data using 3D Perlin noise and outputs as GeoTIFF files.
Creates 4 directories with time-series precipitation data:
- current_precip: instantaneous precipitation intensity (mm/hr)
- total_1h: cumulative precipitation over 1 hour (mm)
- total_2h: cumulative precipitation over 2 hours (mm)
- total_24h: cumulative precipitation over 24 hours (mm)

All files in all four directories are generated from the same storm evolution.
Each frame in a directory represents the accumulated water precipitation in the time interval
of the label of the directory.
"""

import numpy as np
import rasterio
from noise import pnoise3
from pathlib import Path
from collections import deque
import os
import time
from multiprocessing import cpu_count

# --- CONFIGURATION ---
SIMULATION_HOURS = 8  # Simulation duration (hours)
TIME_STEP_MINUTES = 5  # Time step between frames (minutes)
DEMO_SPEED_SECONDS = 0  # Pause between frames for demo (set to 0 for no pause)
MAP_RESOLUTION = 30  # Raster resolution (m/pixel)

# Performance optimization
USE_PARALLEL = True  # Enable parallel processing (highly recommended)
NUM_WORKERS = None  # Number of CPU cores to use (None = auto-detect all cores)
CHUNK_SIZE = 200  # Number of rows to process per chunk (adjust based on RAM)

# Perlin noise parameters
SCALE = 100.0  # Spatial scale for noise
OCTAVES = 6  # Detail level
PERSISTENCE = 0.5  # Amplitude persistence across octaves
LACUNARITY = 2.0  # Frequency multiplier across octaves
TIME_SCALE = 0.1  # Temporal evolution speed

# Precipitation parameters - ENHANCED FOR HEAVY FLOODING
MAX_INTENSITY = 215.0  # Maximum precipitation intensity (mm/hr)
INTENSITY_THRESHOLD = 0.2  # Threshold below which precip is 0 (creates storm cells)
INTENSITY_CONCENTRATION = (
        3.25  # Power exponent for storm concentration (higher = more intense cores)
)

# Storm intensity profile parameters
STORM_PEAK_POSITION = 0.4  # Peak occurs at 40% through the simulation (0.0 to 1.0)
STORM_PEAK_WIDTH = 0.2  # Width of peak intensity period (0.0 to 1.0)
STORM_BUILD_RATE = 2.5  # How quickly storm builds up (higher = faster)
STORM_DECAY_RATE = 1.5  # How quickly storm decays (higher = faster)

# Storm movement parameters
STORM_DIRECTION = (
        45  # Direction storm moves (degrees, 0=East, 90=North, 180=West, 270=South)
)
STORM_SPEED = 0.3  # Storm movement speed (pixels per frame, typical: 0.1-0.5)
WIND_STRENGTH = 0.2  # Wind influence on precipitation (0.0-1.0)
ENABLE_VORTICITY = True  # Enable rotation/tornado effects during peak
VORTICITY_STRENGTH = 0.1  # Strength of rotational effects (0.0-0.2)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Input/Output directories
INPUT_TIF = DATA_DIR / "static" / "Hanoi_LULC_30m_UTM.tif"  # Master raster file
OUTPUT_DIR_CURRENT = DATA_DIR / "storm_generator" / "current_precip"
OUTPUT_DIR_1H = DATA_DIR / "storm_generator" / "total_1h"
OUTPUT_DIR_2H = DATA_DIR / "storm_generator" / "total_2h"
OUTPUT_DIR_24H = DATA_DIR / "storm_generator" / "total_24h"


# --- HELPER FUNCTIONS ---


def create_directories():
        """Creates necessary output directories."""
        for dir_path in [
                OUTPUT_DIR_CURRENT,
                OUTPUT_DIR_1H,
                OUTPUT_DIR_2H,
                OUTPUT_DIR_24H,
        ]:
                os.makedirs(dir_path, exist_ok=True)


def save_geotiff(filepath, data, transform, crs):
        """Saves a NumPy array as a GeoTIFF file."""
        with rasterio.open(
                filepath,
                "w",
                driver="GTiff",
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype="float32",
                crs=crs,
                transform=transform,
                compress="lzw",
        ) as dst:
                dst.write(data, 1)


class StormGenerator:
        """Generate synthetic storm data using 3D Perlin noise."""

        def __init__(self, width, height, transform, crs):
                """
                Initialize the storm generator.

                Args:
                        width: Width of the grid in pixels
                        height: Height of the grid in pixels
                        transform: Rasterio affine transform
                        crs: Coordinate Reference System
                """
                self.width = width
                self.height = height
                self.transform = transform
                self.crs = crs

                # Pre-compute coordinate grids for vectorization
                self.x_grid, self.y_grid = np.meshgrid(
                        np.arange(width, dtype=np.float32),
                        np.arange(height, dtype=np.float32),
                        indexing="xy",
                )

                # Pre-compute storm center
                self.center_x = width / 2
                self.center_y = height / 2

        def _compute_noise_chunk_serial(
                self, y_start, y_end, nx_chunk, ny_chunk, time_z
        ):
                """
                Compute noise for a chunk of rows (optimized serial version).

                Args:
                        y_start: Starting row index
                        y_end: Ending row index
                        nx_chunk: Normalized x coordinates for this chunk only
                        ny_chunk: Normalized y coordinates for this chunk only
                        time_z: Time coordinate for 3D noise

                Returns:
                        Chunk of noise field
                """
                chunk_height = y_end - y_start
                noise_chunk = np.zeros((chunk_height, self.width), dtype=np.float32)

                for y_local in range(chunk_height):
                        for x in range(self.width):
                                noise_val = pnoise3(
                                        nx_chunk[y_local, x] * SCALE,
                                        ny_chunk[y_local, x] * SCALE,
                                        time_z,
                                        octaves=OCTAVES,
                                        persistence=PERSISTENCE,
                                        lacunarity=LACUNARITY,
                                        repeatx=self.width,
                                        repeaty=self.height,
                                        base=0,
                                )
                                noise_chunk[y_local, x] = noise_val

                return noise_chunk

        def generate_noise_field(self, frame: int) -> np.ndarray:
                """
                Generate a 2D noise field for a specific time frame using 3D Perlin noise.
                Includes storm movement, wind effects, and optional vorticity.
                OPTIMIZED: Uses vectorized operations and parallel processing.

                Args:
                        frame: Frame number (time index)

                Returns:
                        2D array of noise values [-1, 1]
                """
                time_z = frame * TIME_SCALE

                # Calculate storm movement offset (vectorized)
                direction_rad = np.radians(STORM_DIRECTION)
                offset_x = frame * STORM_SPEED * np.cos(direction_rad)
                offset_y = frame * STORM_SPEED * np.sin(direction_rad)

                # Vectorized coordinate transformation with storm movement
                nx = (self.x_grid - offset_x) / self.width
                ny = (self.y_grid - offset_y) / self.height

                # Add wind-based distortion (vectorized)
                wind_offset_x = frame * WIND_STRENGTH * np.cos(direction_rad) * 0.1
                wind_offset_y = frame * WIND_STRENGTH * np.sin(direction_rad) * 0.1
                nx += wind_offset_x
                ny += wind_offset_y

                # Add vorticity (rotation) - vectorized
                if ENABLE_VORTICITY:
                        dx = self.x_grid - self.center_x
                        dy = self.y_grid - self.center_y
                        distance = np.sqrt(dx**2 + dy**2)

                        # Rotation strength (exponential decay from center)
                        rotation_strength = VORTICITY_STRENGTH * np.exp(
                                -distance / (max(self.width, self.height) * 0.3)
                        )

                        # Calculate rotation
                        angle = np.arctan2(dy, dx) + rotation_strength * frame
                        rotated_x = self.center_x + distance * np.cos(angle)
                        rotated_y = self.center_y + distance * np.sin(angle)

                        # Blend original and rotated coordinates
                        blend = rotation_strength * 5
                        blend = np.clip(blend, 0, 1)  # Ensure blend stays in [0, 1]
                        nx = nx * (1 - blend) + (rotated_x / self.width) * blend
                        ny = ny * (1 - blend) + (rotated_y / self.height) * blend

                # Generate noise field using chunked parallel processing
                if USE_PARALLEL:
                        noise_field = self._generate_noise_parallel(nx, ny, time_z)
                else:
                        noise_field = self._generate_noise_serial(nx, ny, time_z)

                return noise_field

        def _generate_noise_serial(self, nx, ny, time_z):
                """Generate noise field serially (fallback method)."""
                noise_field = np.zeros((self.height, self.width), dtype=np.float32)

                for y in range(self.height):
                        for x in range(self.width):
                                noise_val = pnoise3(
                                        nx[y, x] * SCALE,
                                        ny[y, x] * SCALE,
                                        time_z,
                                        octaves=OCTAVES,
                                        persistence=PERSISTENCE,
                                        lacunarity=LACUNARITY,
                                        repeatx=self.width,
                                        repeaty=self.height,
                                        base=0,
                                )
                                noise_field[y, x] = noise_val

                return noise_field

        def _generate_noise_parallel(self, nx, ny, time_z):
                """Generate noise field using chunked processing (optimized for Windows)."""
                # Split work into chunks
                chunk_ranges = []
                for i in range(0, self.height, CHUNK_SIZE):
                        y_start = i
                        y_end = min(i + CHUNK_SIZE, self.height)
                        chunk_ranges.append((y_start, y_end))

                # Pre-allocate output array
                noise_field = np.zeros((self.height, self.width), dtype=np.float32)

                # Process chunks sequentially to avoid memory issues
                # Each chunk uses internal parallelization via NumPy's BLAS/LAPACK
                for y_start, y_end in chunk_ranges:
                        # Extract only the chunk we need (avoids passing full arrays)
                        nx_chunk = nx[y_start:y_end, :]
                        ny_chunk = ny[y_start:y_end, :]

                        # Compute this chunk
                        chunk = self._compute_noise_chunk_serial(
                                y_start, y_end, nx_chunk, ny_chunk, time_z
                        )

                        # Store result
                        noise_field[y_start:y_end, :] = chunk

                return noise_field

        def noise_to_precipitation(
                self,
                noise_field: np.ndarray,
                frame: int,
                num_frames: int,
        ) -> np.ndarray:
                """
                Convert noise field to precipitation intensity with realistic storm evolution.
                Storm builds up gradually, reaches peak intensity, then decays.

                Args:
                        noise_field: 2D array of noise values [-1, 1]
                        frame: Current frame number
                        num_frames: Total number of frames

                Returns:
                        2D array of precipitation intensity (mm/hr)
                """
                # Calculate storm intensity multiplier based on temporal position
                progress = frame / num_frames  # 0.0 to 1.0

                # Calculate distance from peak
                distance_from_peak = abs(progress - STORM_PEAK_POSITION)

                # Build-up phase (before peak)
                if progress < STORM_PEAK_POSITION:
                        # Exponential growth towards peak
                        normalized_distance = distance_from_peak / STORM_PEAK_POSITION
                        intensity_multiplier = 1.0 - np.exp(
                                -STORM_BUILD_RATE * (1.0 - normalized_distance)
                        )
                # Decay phase (after peak)
                else:
                        # Exponential decay from peak
                        normalized_distance = distance_from_peak / (
                                1.0 - STORM_PEAK_POSITION
                        )
                        intensity_multiplier = np.exp(
                                -STORM_DECAY_RATE * normalized_distance
                        )

                # Add peak intensity boost (Gaussian-like peak)
                if distance_from_peak < STORM_PEAK_WIDTH:
                        peak_boost = np.exp(
                                -((distance_from_peak / STORM_PEAK_WIDTH) ** 2) * 4
                        )
                        intensity_multiplier = intensity_multiplier * (
                                1.0 + peak_boost * 0.5
                        )

                # Normalize noise to [0, 1]
                normalized = (noise_field + 1.0) / 2.0

                # Apply threshold to create storm cells
                precip = np.where(
                        normalized > INTENSITY_THRESHOLD,
                        (normalized - INTENSITY_THRESHOLD)
                        / (1.0 - INTENSITY_THRESHOLD),
                        0.0,
                )

                # Apply power function for concentration - creates intense cores
                precip_concentrated = np.power(precip, INTENSITY_CONCENTRATION)

                # Scale to actual precipitation intensity with temporal multiplier
                precip_intensity = (
                        precip_concentrated * MAX_INTENSITY * intensity_multiplier
                )

                # Add extreme events: boost the top 10% of values even more during peak periods
                if intensity_multiplier > 0.7:  # Only during strong storm periods
                        percentile_90 = (
                                np.percentile(
                                        precip_intensity[precip_intensity > 0], 90
                                )
                                if np.any(precip_intensity > 0)
                                else 0
                        )
                        extreme_mask = precip_intensity > percentile_90
                        precip_intensity[extreme_mask] *= 1.3  # Boost extreme values

                return precip_intensity.astype(np.float32)


# --- MAIN SIMULATION LOGIC ---


def run_simulation():
        """Main function to run the Perlin noise-based storm simulation."""
        create_directories()

        # Load spatial configuration from input GeoTIFF
        with rasterio.open(INPUT_TIF) as src:
                bounds = src.bounds
                transform = src.transform
                nx, ny = src.width, src.height
                crs = src.crs

        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
        print(
                f"Extracted bounds: west={west}, south={south}, east={east}, north={north}"
        )
        print(f"Grid size: {nx} x {ny} pixels")
        print(f"CRS: {crs}")
        print(f"Map resolution: {MAP_RESOLUTION} m/pixel")

        # Calculate number of frames
        num_frames = int((SIMULATION_HOURS * 60) / TIME_STEP_MINUTES)
        frames_per_hour = 60 // TIME_STEP_MINUTES
        history_size = 24 * frames_per_hour

        print(
                f"\nStarting {SIMULATION_HOURS}-hour simulation with {num_frames} frames..."
        )
        print(f"Time step: {TIME_STEP_MINUTES} minutes")
        print(f"Max intensity: {MAX_INTENSITY} mm/hr")
        print(f"Storm concentration factor: {INTENSITY_CONCENTRATION}")
        print(
                f"Storm peak position: {STORM_PEAK_POSITION * 100:.0f}% through simulation"
        )
        print(
                f"Storm profile: Build rate={STORM_BUILD_RATE}, Decay rate={STORM_DECAY_RATE}"
        )
        print(
                f"Storm movement: Direction={STORM_DIRECTION}°, Speed={STORM_SPEED} px/frame"
        )
        print(f"Wind strength: {WIND_STRENGTH}")
        if ENABLE_VORTICITY:
                print(f"Vorticity enabled: Strength={VORTICITY_STRENGTH}")

        # Performance info
        n_workers = NUM_WORKERS if NUM_WORKERS else cpu_count()
        if USE_PARALLEL:
                print(
                        f"Performance: Optimized chunked processing ({n_workers} CPU cores detected)"
                )
                print(
                        "             Vectorized coordinate transforms + NumPy BLAS acceleration"
                )
        else:
                print("Performance: Serial processing (single core)")
        print(f"Chunk size: {CHUNK_SIZE} rows per batch")

        print(f"Output directory: {OUTPUT_DIR_CURRENT.parent}\n")

        # Initialize storm generator
        generator = StormGenerator(nx, ny, transform, crs)

        # Precipitation history for cumulative calculations
        precip_history = deque(maxlen=history_size)

        # Simulation loop
        start_time = time.time()
        for t in range(num_frames):
                frame_start = time.time()

                # Generate noise field and convert to precipitation
                noise_field = generator.generate_noise_field(t)
                current_precip = generator.noise_to_precipitation(
                        noise_field, t, num_frames
                )

                # Calculate statistics for logging
                max_precip = current_precip.max()
                mean_precip = (
                        current_precip[current_precip > 0].mean()
                        if np.any(current_precip > 0)
                        else 0
                )
                coverage = (current_precip > 0).sum() / (nx * ny) * 100

                # Calculate current storm phase
                progress = t / num_frames
                if progress < STORM_PEAK_POSITION - STORM_PEAK_WIDTH:
                        phase = "BUILD-UP"
                elif progress < STORM_PEAK_POSITION + STORM_PEAK_WIDTH:
                        phase = "PEAK"
                else:
                        phase = "DECAY"

                frame_time = time.time() - frame_start
                elapsed = time.time() - start_time
                avg_time_per_frame = elapsed / (t + 1)
                eta_seconds = avg_time_per_frame * (num_frames - t - 1)
                eta_minutes = int(eta_seconds / 60)
                eta_secs = int(eta_seconds % 60)

                # Log every frame with detailed info
                print(
                        f"Frame {t + 1:3d}/{num_frames} [{phase:8s}] | "
                        f"Max: {max_precip:6.2f} mm/hr | "
                        f"Avg: {mean_precip:6.2f} mm/hr | "
                        f"Coverage: {coverage:5.2f}% | "
                        f"Time: {frame_time:.2f}s | "
                        f"ETA: {eta_minutes}m {eta_secs}s"
                )

                # Store in history
                precip_history.append(current_precip)

                # Save current precipitation intensity
                filename_current = f"current_precip_frame_{t:04d}.tif"
                save_geotiff(
                        os.path.join(OUTPUT_DIR_CURRENT, filename_current),
                        current_precip,
                        transform,
                        crs,
                )

                # Calculate cumulative precipitation for different time windows
                history_array = np.array(precip_history)
                time_fraction = TIME_STEP_MINUTES / 60.0

                # 1-hour total
                filename_1h = f"total_1h_frame_{t:04d}.tif"
                total_1h = (
                        np.sum(history_array[-frames_per_hour:], axis=0) * time_fraction
                )
                save_geotiff(
                        os.path.join(OUTPUT_DIR_1H, filename_1h),
                        total_1h,
                        transform,
                        crs,
                )

                # 2-hour total
                filename_2h = f"total_2h_frame_{t:04d}.tif"
                total_2h = (
                        np.sum(history_array[-(frames_per_hour * 2) :], axis=0)
                        * time_fraction
                )
                save_geotiff(
                        os.path.join(OUTPUT_DIR_2H, filename_2h),
                        total_2h,
                        transform,
                        crs,
                )

                # 24-hour total (calculated from all available history up to the max)
                filename_24h = f"total_24h_frame_{t:04d}.tif"
                total_24h = np.sum(history_array, axis=0) * time_fraction
                save_geotiff(
                        os.path.join(OUTPUT_DIR_24H, filename_24h),
                        total_24h,
                        transform,
                        crs,
                )

                # Optional: pause for demo visualization
                if t < num_frames - 1 and DEMO_SPEED_SECONDS > 0:
                        time.sleep(DEMO_SPEED_SECONDS)

        total_time = time.time() - start_time
        print(
                f"\nStorm generation complete in {total_time:.1f} seconds ({total_time / 60:.1f} minutes)!"
        )
        print(f"Average time per frame: {total_time / num_frames:.2f} seconds")
        print(f"Generated {num_frames} frames in each of 4 directories:")
        print(f"  - current_precip: {OUTPUT_DIR_CURRENT}")
        print(f"  - total_1h: {OUTPUT_DIR_1H}")
        print(f"  - total_2h: {OUTPUT_DIR_2H}")
        print(f"  - total_24h: {OUTPUT_DIR_24H}")


if __name__ == "__main__":
        run_simulation()
