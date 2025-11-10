"""
Storm Generator Script
Generates synthetic precipitation data using Perlin noise and outputs as GeoTIFF files.
Creates 4 directories with time-series precipitation data:
- current_precip: instantaneous precipitation intensity (mm/hr)
- total_1h: cumulative precipitation over 1 hour (mm)
- total_2h: cumulative precipitation over 2 hours (mm)
- total_24h: cumulative precipitation over 24 hours (mm)
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from noise import pnoise3
from pathlib import Path
from typing import Tuple, Optional

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class StormGenerator:
        """Generate synthetic storm data using 3D Perlin noise."""

        def __init__(
                self,
                width: int = 512,
                height: int = 512,
                num_frames: int = 100,
                bounds: Tuple[float, float, float, float] = (-180, -90, 180, 90),
                time_step_minutes: int = 5,
                output_dir: Optional[str] = None,
                pixel_size: Optional[float] = None,
                crs: str = "EPSG:4326",
        ):
                """
                Initialize the storm generator.
                Input:
                    width: Width of the grid in pixels
                    height: Height of the grid in pixels
                    num_frames: Number of time frames to generate
                    bounds: Geographic bounds (west, south, east, north) - coordinates should match the CRS
                    time_step_minutes: Time between frames in minutes
                    output_dir: Base directory for output files (default: {project_root}/data/storm_generator)
                    pixel_size: Pixel size/resolution in the same units as bounds (e.g., degrees or meters).
                                If all three (width, height, pixel_size) are specified: bounds are adjusted
                                If only pixel_size is specified: width/height calculated from bounds
                                If pixel_size + width specified: height calculated, east bound adjusted
                                If pixel_size + height specified: width calculated, north bound adjusted
                                If pixel_size not specified: width/height used directly with given bounds
                    crs: Coordinate Reference System (e.g., 'EPSG:4326' for WGS84 lat/lon,
                         'EPSG:32648' for UTM Zone 48N). Bounds should be in the units of this CRS.
                """
                self.num_frames = num_frames
                self.time_step_minutes = time_step_minutes
                self.output_dir = (
                        Path(output_dir)
                        if output_dir
                        else (DATA_DIR / "storm_generator")
                )
                self.pixel_size = pixel_size
                self.crs = crs  # Store CRS early for use in calculations

                west, south, east, north = bounds

                # Determine width, height, and bounds based on input parameters
                if pixel_size is not None:
                        # Check if both width and height are explicitly specified
                        width_specified = width is not None and width != 512
                        height_specified = height is not None and height != 512

                        if width_specified and height_specified:
                                # All three specified: use width, height, and pixel_size
                                # Adjust both east and north bounds
                                self.width = width
                                self.height = height
                                east = west + (self.width * pixel_size)
                                north = south + (self.height * pixel_size)
                                self.bounds = (west, south, east, north)
                        elif width_specified:
                                # Width and pixel_size specified: calculate height, adjust east bound
                                self.width = width
                                self.height = int(np.ceil((north - south) / pixel_size))
                                east = west + (self.width * pixel_size)
                                self.bounds = (west, south, east, north)
                        elif height_specified:
                                # Height and pixel_size specified: calculate width, adjust north bound
                                self.height = height
                                self.width = int(np.ceil((east - west) / pixel_size))
                                north = south + (self.height * pixel_size)
                                self.bounds = (west, south, east, north)
                        else:
                                # Only pixel_size specified: calculate both from bounds
                                self.width = int(np.ceil((east - west) / pixel_size))
                                self.height = int(np.ceil((north - south) / pixel_size))
                                self.bounds = bounds
                else:
                        # No pixel_size: use specified width and height with given bounds
                        self.width = width
                        self.height = height
                        self.bounds = bounds  # Perlin noise parameters
                self.scale = 100.0  # Spatial scale
                self.octaves = 6  # Detail level
                self.persistence = 0.5
                self.lacunarity = 2.0
                self.time_scale = 0.1  # Temporal evolution speed

                # Precipitation parameters - ENHANCED FOR HEAVY FLOODING
                self.max_intensity = 150.0  # Maximum precipitation intensity (mm/hr) - increased from 50
                self.intensity_threshold = 0.2  # Threshold below which precip is 0 - lowered to create more coverage
                self.intensity_concentration = 2.5  # Power exponent for storm concentration (higher = more intense cores)

                # Output directories
                self.dirs = {
                        "current_precip": self.output_dir / "current_precip",
                        "total_1h": self.output_dir / "total_1h",
                        "total_2h": self.output_dir / "total_2h",
                        "total_24h": self.output_dir / "total_24h",
                }

                for dir_path in self.dirs.values():
                        dir_path.mkdir(parents=True, exist_ok=True)

                # Calculate GeoTIFF transform
                west, south, east, north = self.bounds
                self.transform = from_bounds(
                        west, south, east, north, self.width, self.height
                )

                # Calculate actual pixel size for reference
                self.actual_pixel_width = (east - west) / self.width
                self.actual_pixel_height = (north - south) / self.height

        def generate_noise_field(self, frame: int) -> np.ndarray:
                """
                Generate a 2D noise field for a specific time frame.
                Input: frame: Frame number (time index)
                Output: 2D array of noise values [-1, 1]
                """
                noise_field = np.zeros((self.height, self.width), dtype=np.float32)

                time_z = frame * self.time_scale

                for y in range(self.height):
                        for x in range(self.width):
                                # Normalize coordinates
                                nx = x / self.width
                                ny = y / self.height

                                # Generate 3D Perlin noise (x, y, time)
                                noise_val = pnoise3(
                                        nx * self.scale,
                                        ny * self.scale,
                                        time_z,
                                        octaves=self.octaves,
                                        persistence=self.persistence,
                                        lacunarity=self.lacunarity,
                                        repeatx=self.width,
                                        repeaty=self.height,
                                        base=0,
                                )

                                noise_field[y, x] = noise_val

                return noise_field

        def noise_to_precipitation(self, noise_field: np.ndarray) -> np.ndarray:
                """
                Convert noise field to precipitation intensity.
                Enhanced to create more intense storms with concentrated cores.
                Input: noise_field: 2D array of noise values [-1, 1]
                Output: 2D array of precipitation intensity (mm/hr)
                """
                # Normalize noise to [0, 1]
                normalized = (noise_field + 1.0) / 2.0

                # Apply threshold to create storm cells
                precip = np.where(
                        normalized > self.intensity_threshold,
                        (normalized - self.intensity_threshold)
                        / (1.0 - self.intensity_threshold),
                        0.0,
                )

                # Enhanced: Apply power function for concentration - creates intense cores
                # Higher power = more concentrated intense rainfall in storm centers
                precip_concentrated = np.power(precip, self.intensity_concentration)

                # Scale to actual precipitation intensity
                precip_intensity = precip_concentrated * self.max_intensity

                # Add extreme events: boost the top 10% of values even more
                percentile_90 = (
                        np.percentile(precip_intensity[precip_intensity > 0], 90)
                        if np.any(precip_intensity > 0)
                        else 0
                )
                extreme_mask = precip_intensity > percentile_90
                precip_intensity[extreme_mask] *= 1.5  # Boost extreme values by 50%

                return precip_intensity.astype(np.float32)

        def save_geotiff(
                self,
                data: np.ndarray,
                filepath: Path,
                description: str = "Precipitation",
        ):
                """
                data: 2D array to save
                filepath: Output file path
                description: Description metadata
                """
                with rasterio.open(
                        filepath,
                        "w",
                        driver="GTiff",
                        height=data.shape[0],
                        width=data.shape[1],
                        count=1,
                        dtype=data.dtype,
                        crs=self.crs,
                        transform=self.transform,
                        compress="lzw",
                ) as dst:
                        dst.write(data, 1)
                        dst.set_band_description(1, description)

        def calculate_cumulative(
                self, precip_history: list, window_frames: int
        ) -> np.ndarray:
                """
                Calculate cumulative precipitation over a time window.
                Input:
                    precip_history: List of precipitation intensity arrays
                    window_frames: Number of frames to accumulate
                Output:
                    Cumulative precipitation (mm)
                """
                if len(precip_history) == 0:
                        return np.zeros((self.height, self.width), dtype=np.float32)

                # Take the last N frames
                frames_to_sum = precip_history[-window_frames:]

                # Sum precipitation (intensity * time_step in hours)
                time_step_hours = self.time_step_minutes / 60.0
                cumulative = np.sum(frames_to_sum, axis=0) * time_step_hours

                return cumulative.astype(np.float32)

        def generate_storm(self, verbose: bool = True):
                if verbose:
                        print(f"Generating storm with {self.num_frames} frames...")
                        print(f"Grid size: {self.width}x{self.height}")
                        print(
                                f"Pixel size: {self.actual_pixel_width:.6f} x {self.actual_pixel_height:.6f}"
                        )
                        print(f"Bounds: {self.bounds}")
                        print(f"CRS: {self.crs}")
                        print(f"Time step: {self.time_step_minutes} minutes")
                        print(f"Output directory: {self.output_dir}")
                        print(f"Max intensity: {self.max_intensity} mm/hr")
                        print(
                                f"Storm concentration factor: {self.intensity_concentration}"
                        )

                # Define intense storm events at specific frames
                # These frames will have extra intensity boost
                intense_storm_frames = [
                        int(self.num_frames * 0.25),  # 25% through
                        int(self.num_frames * 0.50),  # 50% through (peak)
                        int(self.num_frames * 0.75),  # 75% through
                ]

                if verbose:
                        print(
                                f"Intense storm events scheduled at frames: {intense_storm_frames}"
                        )

                # Store precipitation history for cumulative calculations
                precip_history = []

                # Calculate window sizes in frames
                frames_1h = int(60 / self.time_step_minutes)
                frames_2h = int(120 / self.time_step_minutes)
                frames_24h = int(1440 / self.time_step_minutes)

                for frame in range(self.num_frames):
                        if verbose and frame % 10 == 0:
                                print(f"Processing frame {frame}/{self.num_frames}...")

                        # Generate noise field and convert to precipitation
                        noise_field = self.generate_noise_field(frame)
                        precip_intensity = self.noise_to_precipitation(noise_field)

                        # Apply intensity boost for extreme storm events
                        if frame in intense_storm_frames:
                                # Create 2x intensity boost with some spatial variation
                                boost_factor = 2.0 + 0.5 * np.random.rand(
                                        *precip_intensity.shape
                                )
                                precip_intensity = precip_intensity * boost_factor
                                # Cap at a reasonable maximum (200 mm/hr is extreme tropical storm level)
                                precip_intensity = np.minimum(precip_intensity, 200.0)
                                if verbose:
                                        print(
                                                f"  ⚡ INTENSE STORM EVENT at frame {frame}! Peak: {precip_intensity.max():.1f} mm/hr"
                                        )

                        # Store in history
                        precip_history.append(precip_intensity)

                        # Save current precipitation intensity
                        filename = f"precip_frame_{frame:04d}.tif"
                        self.save_geotiff(
                                precip_intensity,
                                self.dirs["current_precip"] / filename,
                                "Precipitation intensity (mm/hr)",
                        )

                        # Calculate and save cumulative precipitation
                        # 1-hour total
                        total_1h = self.calculate_cumulative(precip_history, frames_1h)
                        self.save_geotiff(
                                total_1h,
                                self.dirs["total_1h"] / filename,
                                "1-hour cumulative precipitation (mm)",
                        )

                        # 2-hour total
                        total_2h = self.calculate_cumulative(precip_history, frames_2h)
                        self.save_geotiff(
                                total_2h,
                                self.dirs["total_2h"] / filename,
                                "2-hour cumulative precipitation (mm)",
                        )

                        # 24-hour total
                        total_24h = self.calculate_cumulative(
                                precip_history, frames_24h
                        )
                        self.save_geotiff(
                                total_24h,
                                self.dirs["total_24h"] / filename,
                                "24-hour cumulative precipitation (mm)",
                        )

                if verbose:
                        print("\nStorm generation complete!")
                        print(
                                f"Generated {self.num_frames} frames in each of 4 directories:"
                        )
                        for name, path in self.dirs.items():
                                print(f"  - {name}: {path}")


def main():
        # Create generator
        generator = StormGenerator(
                width=2546,
                height=2979,
                num_frames=100,
                bounds=(
                        529245.0790000000270084,
                        2274406.6759999999776483,
                        605625.0790000000270084,
                        2363776.6759999999776483,
                ),
                time_step_minutes=5,
                output_dir=str(DATA_DIR / "storm_generator"),
                pixel_size=30,
                crs="EPSG:32648",
        )

        generator.generate_storm()


if __name__ == "__main__":
        main()
