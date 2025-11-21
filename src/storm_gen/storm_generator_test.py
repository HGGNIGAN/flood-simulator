import numpy as np
from perlin_noise import PerlinNoise
import rasterio
from pathlib import Path
import time
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# --- CONFIGURATION ---
SIMULATION_HOURS = 8  # Simulation duration
TIME_STEP_MINUTES = 15  # Time step
DEMO_SPEED_SECONDS = 1  # Pause between frames for demo
MAP_RESOLUTION = 30  # Raster resolution (m/pixel)

MAX_INTENSITY_MM_HR = 150  # Peak intensity at the eye of the storm
STORM_WIDTH_FACTOR = 2  # Relative width of the storm band
STORM_SPEED = 600  # Speed of the storm (m/min)
STORM_ACCELERATION = -0.5  # Acceleration of the storm (m/min^2)
DRY_THRESHOLD_MM_HR = 20  # Minimum intensity for rain

SCALE = 250.0  # Noise scale
NOISE_EVOLUTION_SPEED = 0.1  # Controls how fast the cloud texture changes over time
NOISE_OCTAVES = 6  # Noise octaves
NOISE_PERSISTENCE = 0.7  # Noise persistence
NOISE_LACUNARITY = 2.0  # Noise lacunarity

# Performance optimization
USE_PARALLEL = True  # Enable parallel noise generation
NUM_WORKERS = None  # Number of CPU cores (None = auto-detect)
CHUNK_SIZE = 100  # Number of rows to process per chunk

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Input/Output directories
INPUT_TIF = DATA_DIR / "static" / "Hanoi_LULC_30m_UTM.tif"  # Master raster file
OUTPUT_DIR_CURRENT = DATA_DIR / "storm_generator_test" / "current_precip"
OUTPUT_DIR_1H = DATA_DIR / "storm_generator_test" / "total_1h"
OUTPUT_DIR_2H = DATA_DIR / "storm_generator_test" / "total_2h"
OUTPUT_DIR_24H = DATA_DIR / "storm_generator_test" / "total_24h"


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
        ) as dst:
                dst.write(data, 1)


def generate_dynamic_noise_chunk(start_row, end_row, width, height, time_step, seed=0):
        """Generate noise for a chunk of rows - optimized for parallel processing."""
        noise_generator = PerlinNoise(octaves=NOISE_OCTAVES, seed=seed)
        time_component = time_step * NOISE_EVOLUTION_SPEED

        chunk_height = end_row - start_row
        noise_chunk = np.zeros((chunk_height, width))

        for i in range(chunk_height):
                actual_row = start_row + i
                for j in range(width):
                        noise_chunk[i, j] = noise_generator.noise(
                                [actual_row / SCALE, j / SCALE, time_component]
                        )

        return start_row, noise_chunk


def generate_dynamic_noise(width, height, time_step):
        """Generates a single frame of evolving 3D Perlin noise - optimized version."""
        noise_map = np.zeros((height, width))

        if USE_PARALLEL and height >= CHUNK_SIZE * 2:
                # Parallel processing for larger images
                from concurrent.futures import ProcessPoolExecutor, as_completed
                from functools import partial

                # Create chunks
                chunks = []
                for start_row in range(0, height, CHUNK_SIZE):
                        end_row = min(start_row + CHUNK_SIZE, height)
                        chunks.append((start_row, end_row))

                # Process chunks in parallel
                worker_func = partial(
                        generate_dynamic_noise_chunk,
                        width=width,
                        height=height,
                        time_step=time_step,
                        seed=0,
                )

                with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                        futures = {
                                executor.submit(worker_func, start, end): (start, end)
                                for start, end in chunks
                        }

                        for future in as_completed(futures):
                                start_row, noise_chunk = future.result()
                                end_row = start_row + noise_chunk.shape[0]
                                noise_map[start_row:end_row, :] = noise_chunk
        else:
                # Sequential processing for smaller images or when parallel is disabled
                noise_generator = PerlinNoise(octaves=NOISE_OCTAVES, seed=0)
                time_component = time_step * NOISE_EVOLUTION_SPEED

                for i in range(height):
                        for j in range(width):
                                noise_map[i, j] = noise_generator.noise(
                                        [i / SCALE, j / SCALE, time_component]
                                )

        return noise_map


# --- MAIN SIMULATION LOGIC ---
def run_simulation():
        """Main function to run the storm simulation - optimized version."""
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
        print(f"{nx} x {ny} pixels")

        # Calculate number of frames
        num_frames = int((SIMULATION_HOURS * 60) / TIME_STEP_MINUTES)
        frames_per_hour = 60 // TIME_STEP_MINUTES
        history_size = 24 * frames_per_hour
        precip_history = deque(maxlen=history_size)
        diminish_start_frame = 4 * frames_per_hour

        # Pre-calculate coordinate grids (vectorized - done once)
        x_coords = np.arange(nx)
        y_coords = np.arange(ny)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Pre-calculate storm radius
        storm_radius = nx * STORM_WIDTH_FACTOR

        # Pre-calculate time fraction
        time_fraction = TIME_STEP_MINUTES / 60.0

        # Simulation loop
        print(
                f"\nStarting {SIMULATION_HOURS}-hour simulation with {num_frames} frames..."
        )

        start_time = time.time()

        for t in range(num_frames):
                frame_start = time.time()
                print(f"\n--- Frame {t + 1}/{num_frames} ---")

                # 1. Generate Dynamic Noise (optimized with parallel processing)
                dynamic_noise = generate_dynamic_noise(nx, ny, t)

                # 2. Move the Storm (Fixed Physics)
                delta_t = t * TIME_STEP_MINUTES

                # Calculate Speed in Meters/Minute
                current_speed_m_min = STORM_SPEED + 0.5 * STORM_ACCELERATION * delta_t

                # Convert Meters to Pixels
                current_speed_px_min = current_speed_m_min / MAP_RESOLUTION

                # Calculate Center X (starts at 2.8 * width and moves left)
                storm_center_x = 2.8 * nx - delta_t * current_speed_px_min
                storm_center_y = ny / 2

                # DEBUG: Print location
                print(f"   Storm Center X: {storm_center_x:.0f} (Map Width: {nx})")
                if 0 <= storm_center_x <= nx:
                        print("   STATUS: Storm is ON THE MAP 🟢")
                else:
                        print("   STATUS: Storm is OUTSIDE the map 🔴")

                # 3. Calculate distance for the circular shape (vectorized)
                distance = np.sqrt(
                        (x_grid - storm_center_x) ** 2 + (y_grid - storm_center_y) ** 2
                )

                # 4. Create the circular profile (vectorized)
                storm_profile = 1 - (distance / storm_radius) ** 2
                storm_profile = np.clip(storm_profile, 0, 1)

                # 5. Set intensity (The storm intensity will diminish after 4 hours)
                if t < diminish_start_frame:
                        eye_intensity = MAX_INTENSITY_MM_HR
                else:
                        if num_frames > diminish_start_frame:
                                progress_into_fade = t - diminish_start_frame
                                fade_duration = num_frames - diminish_start_frame
                                fade_factor = 1.0 - (progress_into_fade / fade_duration)
                        else: 
                                fade_factor = 1.0

                        eye_intensity = MAX_INTENSITY_MM_HR * fade_factor

                # 6. Combine with the dynamic noise texture (vectorized)
                noise_texture = (dynamic_noise + 1) / 2
                current_precip_intensity = storm_profile * noise_texture * eye_intensity

                # Apply threshold (vectorized)
                current_precip_intensity[
                        current_precip_intensity < DRY_THRESHOLD_MM_HR
                ] = 0

                # DEBUG: Check if we actually have rain
                max_rain = np.max(current_precip_intensity)
                print(f"   Max Rain Intensity: {max_rain:.2f} mm/h")

                # 7. History & Accumulation Logic
                precip_history.append(current_precip_intensity)

                history_array = np.array(precip_history)

                # Calculate all accumulations (vectorized operations)
                # 1-Hour Total (Accumulation)
                total_1h = (
                        np.sum(history_array[-frames_per_hour:], axis=0) * time_fraction
                )
                save_geotiff(
                        os.path.join(OUTPUT_DIR_1H, f"total_1h_{t:02d}.tif"),
                        total_1h,
                        transform,
                        crs,
                )

                # 2-Hour Total (Accumulation)
                total_2h = (
                        np.sum(history_array[-(frames_per_hour * 2) :], axis=0)
                        * time_fraction
                )
                save_geotiff(
                        os.path.join(OUTPUT_DIR_2H, f"total_2h_{t:02d}.tif"),
                        total_2h,
                        transform,
                        crs,
                )

                # 24-Hour Total (Accumulation)
                total_24h = np.sum(history_array, axis=0) * time_fraction
                save_geotiff(
                        os.path.join(OUTPUT_DIR_24H, f"total_24h_{t:02d}.tif"),
                        total_24h,
                        transform,
                        crs,
                )

                # Save Current Frame (Intensity)
                save_geotiff(
                        os.path.join(OUTPUT_DIR_CURRENT, f"precip_frame_{t:02d}.tif"),
                        current_precip_intensity,
                        transform,
                        crs,
                )

                frame_time = time.time() - frame_start
                print(f"   Frame processing time: {frame_time:.2f}s")

                if t < num_frames - 1:
                        time.sleep(DEMO_SPEED_SECONDS)

        total_time = time.time() - start_time
        avg_time_per_frame = total_time / num_frames
        print("\nSimulation complete.")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per frame: {avg_time_per_frame:.2f}s")


if __name__ == "__main__":
        run_simulation()
