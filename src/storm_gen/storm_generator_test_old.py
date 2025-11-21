import numpy as np
from perlin_noise import PerlinNoise
import rasterio
from pathlib import Path
import time
import os
from collections import deque

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


def generate_dynamic_noise(width, height, time_step):
        """Generates a single frame of evolving 3D Perlin noise."""
        noise_map = np.zeros((height, width))
        # Use the time_step to create a unique "slice" of the 3D noise space
        time_component = time_step * NOISE_EVOLUTION_SPEED

        # Create PerlinNoise instance
        noise_generator = PerlinNoise(octaves=NOISE_OCTAVES, seed=0)

        for i in range(height):
                for j in range(width):
                        # perlin_noise uses normalized coordinates [0, 1]
                        noise_map[i, j] = noise_generator.noise(
                                [i / SCALE, j / SCALE, time_component]
                        )
        return noise_map


# --- MAIN SIMULATION LOGIC ---
def run_simulation():
        """Main function to run the storm simulation."""
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
        print(f"{crs}")

        # Calculate number of frames
        num_frames = int((SIMULATION_HOURS * 60) / TIME_STEP_MINUTES)
        frames_per_hour = 60 // TIME_STEP_MINUTES
        history_size = 24 * frames_per_hour
        precip_history = deque(maxlen=history_size)
        diminish_start_frame = 4 * frames_per_hour

        # Simulation loop
        print(
                f"\nStarting {SIMULATION_HOURS}-hour simulation with {num_frames} frames..."
        )
        for t in range(num_frames):
                # --- CHANGE: Call the new function to get the evolving noise map ---
                print(f"Generating dynamic noise for frame {t + 1}/{num_frames}...")
                dynamic_noise = generate_dynamic_noise(nx, ny, t)

                # 1. Define the storm's center point
                # (Phương trình chuyển động thẳng biến đổi đều để mô phỏng chuyển động của bão, có thể thay bằng phương trình chuyển động khác)
                delta_t = t * TIME_STEP_MINUTES
                current_speed = STORM_SPEED + 0.5 * STORM_ACCELERATION * delta_t
                storm_center_x = 2.8 * nx - delta_t * current_speed
                storm_center_y = ny / 2

                # 2. Create coordinate grids
                x_coords = np.arange(nx)
                y_coords = np.arange(ny)
                x_grid, y_grid = np.meshgrid(x_coords, y_coords)

                # 3. Calculate distance for the circular shape
                distance = np.sqrt(
                        (x_grid - storm_center_x) ** 2 + (y_grid - storm_center_y) ** 2
                )

                # 4. Create the circular profile
                storm_radius = nx * STORM_WIDTH_FACTOR
                storm_profile = 1 - (distance / storm_radius) ** 2
                storm_profile = np.clip(storm_profile, 0, 1)

                # Set intensity (The storm intensity will diminish after 4 hours)
                if t < diminish_start_frame:
                        eye_intensity = MAX_INTENSITY_MM_HR
                else:
                        # Storm is past 4 hours and is now weakening.
                        # Calculate a fade-out factor that goes from 1.0 down to 0.0
                        # over the remaining duration of the simulation.

                        # Check to prevent division by zero if simulation ends exactly at 4 hours
                        if num_frames > diminish_start_frame:
                                progress_into_fade = t - diminish_start_frame
                                fade_duration = num_frames - diminish_start_frame
                                fade_factor = 1.0 - (progress_into_fade / fade_duration)
                        else:
                                fade_factor = 1.0  # Will only happen if simulation is exactly 4 hours

                        eye_intensity = MAX_INTENSITY_MM_HR * fade_factor

                # Combine with the dynamic noise texture
                noise_texture = (dynamic_noise + 1) / 2
                current_precip = storm_profile * noise_texture * eye_intensity
                current_precip[current_precip < DRY_THRESHOLD_MM_HR] = 0

                # --- The rest of the saving and history logic remains exactly the same ---
                precip_history.append(current_precip)
                history_array = np.array(precip_history)
                time_fraction = TIME_STEP_MINUTES / 60.0

                # 1-Hour Total: Sums up to the last 4 frames from history.
                total_1h = (
                        np.sum(history_array[-frames_per_hour:], axis=0) * time_fraction
                )
                save_geotiff(
                        os.path.join(OUTPUT_DIR_1H, f"total_1h_frame_{t:04d}.tif"),
                        total_1h,
                        transform,
                        crs,
                )

                # 2-Hour Total: Sums up to the last 8 frames from history.
                total_2h = (
                        np.sum(history_array[-(frames_per_hour * 2) :], axis=0)
                        * time_fraction
                )
                save_geotiff(
                        os.path.join(OUTPUT_DIR_2H, f"total_2h_frame_{t:04d}.tif"),
                        total_2h,
                        transform,
                        crs,
                )

                # 24-Hour Total (always calculated from all available history up to the max)
                total_24h = np.sum(history_array, axis=0) * time_fraction
                save_geotiff(
                        os.path.join(OUTPUT_DIR_24H, f"total_24h_frame_{t:04d}.tif"),
                        total_24h,
                        transform,
                        crs,
                )

                save_geotiff(
                        os.path.join(
                                OUTPUT_DIR_CURRENT, f"current_precip_frame_{t:04d}.tif"
                        ),
                        current_precip,
                        transform,
                        crs,
                )

                print(f"Saved frame {t + 1}/{num_frames}")
                if t < num_frames - 1:
                        time.sleep(DEMO_SPEED_SECONDS)

        print("\nSimulation complete.")


if __name__ == "__main__":
        run_simulation()  # Replace with your file path
