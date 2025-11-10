"""
- Each row represents a geographic location (x, y coordinates)
- Columns: x, y, frame_0, frame_1, ..., frame_N
- Coordinates are in the CRS of the source raster (or WGS84 if converted)
- Values: flood intensity (0-3)
  - 0: No flooding
  - 1: Light flooding (0.05-0.25m)
  - 2: Moderate flooding (0.25-0.40m)
  - 3: Severe flooding (>0.40m)
"""

import os
import glob
import gc
import numpy as np
import rasterio
from rasterio.transform import xy
from pathlib import Path
from typing import List, Tuple
import argparse

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class FloodTimeSeriesExtractor:
        def __init__(
                self,
                input_dir: str,
                output_csv: str,
                pattern: str = "flood_classified_*.tif",
                sample_rate: int = 1,
                chunk_size: int = 20000,
        ):
                """
                input_dir: Directory containing classified flood GeoTIFF files
                output_csv: Output CSV file path
                pattern: File pattern to match (default: flood_classified_*.tif)
                sample_rate: Spatial sampling rate (1 = all pixels, 2 = every other pixel, etc.)
                chunk_size: Number of pixels to write at once (default: 20000 for 32GB RAM)
                """
                self.input_dir = Path(input_dir)
                self.output_csv = Path(output_csv)
                self.pattern = pattern
                self.sample_rate = sample_rate
                self.chunk_size = chunk_size

        def get_frame_files(self) -> List[Tuple[int, Path]]:
                """Get list of classified flood files sorted by frame number."""
                pattern_path = self.input_dir / self.pattern
                files = glob.glob(str(pattern_path))

                if not files:
                        raise ValueError(
                                f"No files found matching pattern: {pattern_path}"
                        )

                # Extract frame numbers and sort
                frame_files = []
                for filepath in files:
                        basename = os.path.basename(filepath)
                        frame_str = basename.split("_")[-1].replace(".tif", "")
                        try:
                                frame_num = int(frame_str)
                                frame_files.append((frame_num, Path(filepath)))
                        except ValueError:
                                print(
                                        f"Warning: Could not parse frame number from {basename}"
                                )

                frame_files.sort(key=lambda x: x[0])
                print(f"Found {len(frame_files)} classified flood files")
                return frame_files

        def read_flood_data(self, filepath: Path) -> Tuple[np.ndarray, dict]:
                src = None
                try:
                        src = rasterio.open(filepath)
                        data = src.read(
                                1
                        ).copy()  # Copy to avoid holding file reference
                        metadata = {
                                "crs": src.crs,
                                "transform": src.transform,
                                "width": src.width,
                                "height": src.height,
                                "nodata": src.nodata,
                        }
                        return data, metadata
                finally:
                        # Ensure file is closed even if error occurs
                        if src is not None:
                                src.close()
                                del src
                        gc.collect()

        def extract_timeseries_frame_by_frame(
                self,
                valid_pixels_only: bool = True,
                min_flood_events: int = 0,
        ) -> None:
                """
                valid_pixels_only: Only include pixels with valid data (not nodata)
                min_flood_events: Minimum number of flood events required to include pixel
                """
                print("\n" + "=" * 80)
                print("EXTRACTING FLOOD TIME SERIES DATA (FRAME-BY-FRAME)")
                print("=" * 80)

                # Get sorted list of frame files
                frame_files = self.get_frame_files()

                if not frame_files:
                        raise ValueError("No frame files found")

                num_frames = len(frame_files)
                print(f"Total frames to process: {num_frames}")

                # Read first frame to get dimensions and metadata
                print("\nReading metadata from first frame...")
                first_frame_num, first_file = frame_files[0]
                first_data, metadata = self.read_flood_data(first_file)

                height, width = first_data.shape
                nodata = metadata["nodata"] if metadata["nodata"] is not None else -9999

                print(f"Raster dimensions: {height} x {width} pixels")
                print(f"Sample rate: 1:{self.sample_rate}")

                # Generate coordinates for all pixels using np.indices (same as data_rasterizer.py)
                print("\nGenerating pixel locations and coordinates...")

                # Create row and column indices
                rows, cols = np.indices((height, width))

                # Apply sampling if needed
                if self.sample_rate > 1:
                        rows = rows[:: self.sample_rate, :: self.sample_rate]
                        cols = cols[:: self.sample_rate, :: self.sample_rate]
                        sampled_data = first_data[
                                :: self.sample_rate, :: self.sample_rate
                        ]
                else:
                        sampled_data = first_data

                # Flatten the arrays
                rows_flat = rows.flatten()
                cols_flat = cols.flatten()

                # Get coordinates for all pixels at once (same as data_rasterizer.py)
                xs, ys = xy(
                        metadata["transform"], rows_flat, cols_flat, offset="center"
                )

                # Convert to arrays
                x_coords = np.array(xs)
                y_coords = np.array(ys)
                sampled_data_flat = sampled_data.flatten()

                # Filter out nodata pixels if requested
                if valid_pixels_only:
                        valid_mask = sampled_data_flat != nodata
                        rows_flat = rows_flat[valid_mask]
                        cols_flat = cols_flat[valid_mask]
                        x_coords = x_coords[valid_mask]
                        y_coords = y_coords[valid_mask]

                # Build pixel_locations list
                pixel_locations = list(zip(rows_flat, cols_flat, x_coords, y_coords))

                total_pixels = len(pixel_locations)
                print(f"Total valid locations: {total_pixels}")

                # Calculate coordinate bounds
                if total_pixels > 0:
                        x_coords = [loc[2] for loc in pixel_locations]
                        y_coords = [loc[3] for loc in pixel_locations]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        print("\n" + "=" * 80)
                        print("COORDINATE SYSTEM AND BOUNDS")
                        print("=" * 80)
                        print(f"CRS: {metadata['crs']}")
                        print("Coordinate bounds:")
                        print(f"  x:  {x_min:.6f} to {x_max:.6f}")
                        print(f"  y:  {y_min:.6f} to {y_max:.6f}")
                        print(
                                f"  Coverage area: {(x_max - x_min):.6f} × {(y_max - y_min):.6f} units"
                        )
                        print("=" * 80)

                        # Confirmation prompt
                        print(f"\nOutput file: {self.output_csv}")
                        print(f"Total locations to export: {total_pixels}")
                        print(f"Total frames: {num_frames}")
                        estimated_size_mb = (total_pixels * num_frames * 2) / (
                                1024 * 1024
                        )  # Rough estimate
                        print(f"Estimated file size: ~{estimated_size_mb:.2f} MB")

                        response = input("\nProceed? (y/n): ").strip().lower()
                        if response != "y":
                                print("Extraction cancelled.")
                                return

                # Estimate memory usage
                estimated_mb = (total_pixels * num_frames * 4) / (1024 * 1024)
                print(f"\nEstimated memory usage: {estimated_mb:.2f} MB")

                # Initialize time series matrix (pixels x frames)
                # Using int8 since values are 0-3
                print("\nAllocating time series matrix...")
                timeseries_matrix = np.full(
                        (total_pixels, num_frames), -1, dtype=np.int8
                )

                # Clear first frame data from memory
                del first_data
                gc.collect()

                # Process each frame sequentially
                print("\nProcessing frames...")
                for frame_idx, (frame_num, filepath) in enumerate(frame_files):
                        print(
                                f"  Frame {frame_idx + 1}/{num_frames} (#{frame_num:04d}): {filepath.name}",
                                end="",
                                flush=True,
                        )

                        # Read current frame
                        frame_data, _ = self.read_flood_data(filepath)

                        # Extract values for all pixels in this frame
                        for pixel_idx, (row, col, x, y) in enumerate(pixel_locations):
                                value = frame_data[row, col]

                                # Store value (replace nodata with -1)
                                if value == nodata or value < 0:
                                        timeseries_matrix[pixel_idx, frame_idx] = -1
                                else:
                                        timeseries_matrix[pixel_idx, frame_idx] = int(
                                                value
                                        )

                        # CRITICAL: Close and delete frame data before next iteration
                        del frame_data
                        gc.collect()

                        print(" ✓")

                print("\n" + "-" * 80)
                print("All frames processed. Writing to CSV...")

                # Prepare CSV header - coordinates only
                header_cols = ["x", "y"]

                # Add frame columns
                for frame_num, _ in frame_files:
                        header_cols.append(f"frame_{frame_num:04d}")

                # Ensure output directory exists
                self.output_csv.parent.mkdir(parents=True, exist_ok=True)

                # Statistics tracking
                stats = {
                        "total_written": 0,
                        "flood_counts": np.zeros(4, dtype=np.int64),
                        "pixels_with_flood": 0,
                }

                # Write to CSV with chunked output for better I/O performance
                print(f"Writing to: {self.output_csv}")
                with open(
                        self.output_csv, "w", buffering=8192 * 1024
                ) as f:  # 8MB buffer
                        # Write header
                        f.write(",".join(header_cols) + "\n")

                        # Process in chunks for efficient writing
                        for chunk_start in range(0, total_pixels, self.chunk_size):
                                chunk_end = min(
                                        chunk_start + self.chunk_size, total_pixels
                                )

                                print(
                                        f"  Writing pixels {chunk_start}-{chunk_end - 1}...",
                                        end="\r",
                                )

                                # Build chunk lines
                                chunk_lines = []
                                for pixel_idx in range(chunk_start, chunk_end):
                                        row, col, x, y = pixel_locations[pixel_idx]

                                        # Get time series for this pixel
                                        pixel_timeseries = timeseries_matrix[
                                                pixel_idx, :
                                        ]

                                        # Check minimum flood events
                                        if min_flood_events > 0:
                                                flood_count = np.sum(
                                                        pixel_timeseries > 0
                                                )
                                                if flood_count < min_flood_events:
                                                        continue

                                        # Build CSV row - coordinates only
                                        csv_row = [x, y]

                                        # Add frame values (replace -1 with empty string)
                                        for value in pixel_timeseries:
                                                if value == -1:
                                                        csv_row.append("")
                                                else:
                                                        csv_row.append(value)
                                                        stats["flood_counts"][
                                                                value
                                                        ] += 1

                                        # Track if pixel has any flooding
                                        if np.any(pixel_timeseries > 0):
                                                stats["pixels_with_flood"] += 1

                                        chunk_lines.append(
                                                ",".join(str(v) for v in csv_row) + "\n"
                                        )
                                        stats["total_written"] += 1

                                # Write chunk at once
                                f.writelines(chunk_lines)

                                # Clear chunk memory
                                del chunk_lines
                                gc.collect()

                # Clear time series matrix
                del timeseries_matrix
                gc.collect()

                print("\n\n" + "-" * 80)
                print("STATISTICS")
                print("-" * 80)
                print(f"Total pixels written: {stats['total_written']}")
                print(f"Total time frames: {num_frames}")

                # Flood intensity distribution
                total_values = stats["flood_counts"].sum()
                if total_values > 0:
                        print("\nFlood intensity distribution:")
                        labels = ["No flood", "Light", "Moderate", "Severe"]
                        for intensity in range(4):
                                count = stats["flood_counts"][intensity]
                                percentage = count / total_values * 100
                                print(
                                        f"  {intensity} ({labels[intensity]:10s}): {count:10d} ({percentage:5.2f}%)"
                                )

                        print(
                                f"\nPixels with flooding: {stats['pixels_with_flood']} ({stats['pixels_with_flood'] / stats['total_written'] * 100:.2f}%)"
                        )

                file_size_mb = self.output_csv.stat().st_size / (1024 * 1024)
                print(f"\nFile size: {file_size_mb:.2f} MB")
                print("=" * 80)
                print("✓ Extraction completed.")


def main():
        parser = argparse.ArgumentParser(
                description="Extract flood classification time series to CSV (memory-efficient)"
        )
        parser.add_argument(
                "--input-dir",
                type=str,
                default=str(DATA_DIR / "simulation_output"),
                help="Directory containing classified flood GeoTIFF files",
        )
        parser.add_argument(
                "--output-csv",
                type=str,
                default=str(DATA_DIR / "flood_timeseries.csv"),
                help="Output CSV file path",
        )
        parser.add_argument(
                "--pattern",
                type=str,
                default="flood_classified_*.tif",
                help="File pattern to match",
        )
        parser.add_argument(
                "--sample-rate",
                type=int,
                default=1,
                help="Spatial sampling rate (1=all pixels, 2=every other pixel, etc.)",
        )
        parser.add_argument(
                "--chunk-size",
                type=int,
                default=20000,
                help="Number of pixels to write at once (default 20000 optimized for 32GB RAM)",
        )
        parser.add_argument(
                "--valid-only",
                action="store_true",
                help="Only include pixels with valid data (not nodata)",
        )
        parser.add_argument(
                "--min-flood-events",
                type=int,
                default=0,
                help="Minimum number of flood events required to include pixel",
        )

        args = parser.parse_args()

        # Create extractor (optimized for 32GB RAM)
        extractor = FloodTimeSeriesExtractor(
                input_dir=args.input_dir,
                output_csv=args.output_csv,
                pattern=args.pattern,
                sample_rate=args.sample_rate,
                chunk_size=args.chunk_size,
        )

        # Extract time series with frame-by-frame processing (OPTIMIZED)
        extractor.extract_timeseries_frame_by_frame(
                valid_pixels_only=args.valid_only,
                min_flood_events=args.min_flood_events,
        )

        print("\nNext steps:")
        print("  1. Load the CSV file in your ML framework")
        print("  2. Use x/y as spatial identifiers for geospatial models")
        print("  3. Use frame columns as features for time series prediction")
        print("  4. Consider using sliding window approach for sequence modeling")


if __name__ == "__main__":
        main()
