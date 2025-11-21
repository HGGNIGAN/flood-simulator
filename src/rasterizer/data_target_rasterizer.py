"""
Flood Time Series Extractor
Extracts flood classification time series data from GeoTIFF files to CSV format.

Output CSV structure:
- Each row represents a geographic location (x, y coordinates)
- Columns: x, y, frame_0, frame_1, ..., frame_N
- Coordinates are in the CRS of the source raster (or WGS84 if converted)
- Values: flood intensity (0-3)
  - 0: No flooding
  - 1: Light flooding (0.05-0.25m)
  - 2: Moderate flooding (0.25-0.40m)
  - 3: Severe flooding (>0.40m)

Performance Optimization Features:
- Parallel frame processing using multiprocessing.Pool (USE_PARALLEL=True)
- Vectorized pixel extraction with NumPy array indexing (USE_VECTORIZATION=True)
- Configurable CPU core count (NUM_WORKERS, default: auto-detect)
- Chunked CSV writing with 8MB buffer for faster I/O
- Memory-efficient processing with aggressive garbage collection
- Fallback to sequential processing for small datasets or if parallel disabled

Configuration:
- Adjust USE_PARALLEL to enable/disable parallel processing
- Set NUM_WORKERS to control CPU usage (None = use all cores)
- USE_VECTORIZATION enables fast NumPy-based pixel extraction
- DEFAULT_CHUNK_SIZE controls memory usage vs write performance trade-off
"""

import os
import glob
import gc
import numpy as np
import rasterio
from rasterio.transform import xy
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

# Try to import numba for JIT acceleration (optional but recommended)
try:
        from numba import jit, prange

        NUMBA_AVAILABLE = True
except ImportError:
        print("    [WARNING] Numba not available. Install with: pip install numba")
        print("    [WARNING] Extraction will run without JIT acceleration (slower)")
        NUMBA_AVAILABLE = False

        # Dummy decorators if numba not available
        def jit(*args, **kwargs):
                def decorator(func):
                        return func

                return decorator

        prange = range

# --- CONFIGURATION ---

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Default input/output paths
DEFAULT_INPUT_DIR = str(DATA_DIR / "simulation_output")
DEFAULT_INPUT_TEST_DIR = str(DATA_DIR / "simulation_output_test")
DEFAULT_OUTPUT_CSV = str(DATA_DIR / "target_data.csv")
DEFAULT_OUTPUT_TEST_CSV = str(DATA_DIR / "target_data_test.csv")

# File matching pattern
DEFAULT_FILE_PATTERN = "flood_classified_*.tif"

# Sampling and chunking parameters
DEFAULT_SAMPLE_RATE = 1  # Spatial sampling (1=all pixels, 2=every other, etc.)
DEFAULT_CHUNK_SIZE = 80000  # Pixels to write at once (adjust based on RAM)
DEFAULT_BUFFER_SIZE = 8192 * 1024  # File I/O buffer size (8MB)

# Performance optimization
USE_PARALLEL = False  # Disable parallel (causes overhead on Windows)
NUM_WORKERS = None  # Number of workers (None = auto-detect CPU cores)
USE_VECTORIZATION = True  # Enable vectorized pixel extraction (faster)

# Data filtering
DEFAULT_VALID_ONLY = True  # Only include valid pixels (not nodata)
DEFAULT_MIN_FLOOD_EVENTS = 0  # Min flood events to include pixel

# Flood classification values
NODATA_VALUE = -1  # Value to use for nodata in output
FLOOD_INTENSITY_LABELS = ["No flood", "Light", "Moderate", "Severe"]  # 0, 1, 2, 3
NUM_FLOOD_CLASSES = 4  # Number of flood intensity classes

# Memory management
ENABLE_AGGRESSIVE_GC = True  # Enable aggressive garbage collection
GC_COLLECT_FREQUENCY = 10  # Collect garbage every N frames (higher = less overhead)

# Coordinate system
COORDINATE_OFFSET = "center"  # Pixel coordinate reference point ("center" or "ul")

# Output formatting
CSV_EMPTY_VALUE = ""  # Value to use for missing/nodata in CSV
COORDINATE_PRECISION = 6  # Decimal places for coordinates

# Progress display
SHOW_PROGRESS = True  # Show progress during processing
PROGRESS_UPDATE_INTERVAL = 10  # Update progress every N% (for percentage display)


# --- MODULE-LEVEL WORKER FUNCTION (for multiprocessing) ---


def _extract_frame_worker(
        frame_info: tuple, pixel_locations: list, nodata: float
) -> Tuple[int, np.ndarray]:
        """
        Module-level worker function to extract pixel values from a single frame.
        Must be at module level for Windows multiprocessing pickle compatibility.
        Returns (frame_idx, values_array).
        """
        frame_idx, frame_num, filepath = frame_info

        # Read frame data directly
        with rasterio.open(filepath) as src:
                frame_data = src.read(1)

        # Preallocate array for pixel values
        num_pixels = len(pixel_locations)
        values = np.full(num_pixels, NODATA_VALUE, dtype=np.int8)

        if USE_VECTORIZATION:
                # Vectorized extraction (much faster)
                rows = np.array([loc[0] for loc in pixel_locations], dtype=np.int32)
                cols = np.array([loc[1] for loc in pixel_locations], dtype=np.int32)

                # Extract all values at once
                extracted_values = frame_data[rows, cols]

                # Replace nodata with NODATA_VALUE
                valid_mask = (extracted_values != nodata) & (extracted_values >= 0)
                values[valid_mask] = extracted_values[valid_mask].astype(np.int8)
        else:
                # Sequential extraction (fallback)
                for pixel_idx, (row, col, _, _) in enumerate(pixel_locations):
                        value = frame_data[row, col]
                        if value != nodata and value >= 0:
                                values[pixel_idx] = int(value)

        # Clean up
        del frame_data

        return frame_idx, values


class FloodTimeSeriesExtractor:
        def __init__(
                self,
                input_dir: Optional[str] = None,
                output_csv: Optional[str] = None,
                pattern: str = DEFAULT_FILE_PATTERN,
                sample_rate: int = DEFAULT_SAMPLE_RATE,
                chunk_size: int = DEFAULT_CHUNK_SIZE,
                use_test: bool = False,
        ):
                """
                Initialize the flood time series extractor.

                Args:
                        input_dir: Directory containing classified flood GeoTIFF files (None = auto-select based on use_test)
                        output_csv: Output CSV file path (None = auto-select based on use_test)
                        pattern: File pattern to match (default: flood_classified_*.tif)
                        sample_rate: Spatial sampling rate (1 = all pixels, 2 = every other pixel, etc.)
                        chunk_size: Number of pixels to write at once (adjust based on RAM)
                        use_test: Whether to use test data directories (default: False for train data)
                """
                # Auto-select directories based on use_test flag
                if input_dir is None:
                        input_dir = (
                                DEFAULT_INPUT_TEST_DIR
                                if use_test
                                else DEFAULT_INPUT_DIR
                        )
                if output_csv is None:
                        output_csv = (
                                DEFAULT_OUTPUT_TEST_CSV
                                if use_test
                                else DEFAULT_OUTPUT_CSV
                        )

                self.input_dir = Path(input_dir)
                self.output_csv = Path(output_csv)
                self.pattern = pattern
                self.sample_rate = sample_rate
                self.chunk_size = chunk_size
                self.use_test = use_test

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
                        if ENABLE_AGGRESSIVE_GC:
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
                        metadata["transform"],
                        rows_flat,
                        cols_flat,
                        offset=COORDINATE_OFFSET,
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
                        print(
                                f"  x:  {x_min:.{COORDINATE_PRECISION}f} to {x_max:.{COORDINATE_PRECISION}f}"
                        )
                        print(
                                f"  y:  {y_min:.{COORDINATE_PRECISION}f} to {y_max:.{COORDINATE_PRECISION}f}"
                        )
                        print(
                                f"  Coverage area: {(x_max - x_min):.{COORDINATE_PRECISION}f} × {(y_max - y_min):.{COORDINATE_PRECISION}f} units"
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
                        (total_pixels, num_frames), NODATA_VALUE, dtype=np.int8
                )

                # Clear first frame data from memory
                del first_data
                if ENABLE_AGGRESSIVE_GC:
                        gc.collect()

                # Pre-compute row/col arrays for vectorized extraction (huge speedup)
                rows_array = None
                cols_array = None
                if USE_VECTORIZATION:
                        rows_array = np.array(
                                [loc[0] for loc in pixel_locations], dtype=np.int32
                        )
                        cols_array = np.array(
                                [loc[1] for loc in pixel_locations], dtype=np.int32
                        )
                        print(f"Pre-computed lookup arrays for {total_pixels} pixels")

                # Process frames
                print("\nProcessing frames...")

                if USE_PARALLEL and num_frames > 2:
                        # Parallel frame processing
                        num_workers = NUM_WORKERS if NUM_WORKERS else cpu_count()
                        print(f"Using {num_workers} CPU cores for parallel processing")

                        # Prepare frame information tuples
                        frame_info_list = [
                                (frame_idx, frame_num, filepath)
                                for frame_idx, (frame_num, filepath) in enumerate(
                                        frame_files
                                )
                        ]

                        # Create partial function with fixed pixel_locations and nodata
                        worker_func = partial(
                                _extract_frame_worker,
                                pixel_locations=pixel_locations,
                                nodata=nodata,
                        )

                        # Process frames in parallel
                        with Pool(processes=num_workers) as pool:
                                results = pool.map(worker_func, frame_info_list)

                        # Populate timeseries matrix from results
                        for frame_idx, values in results:
                                timeseries_matrix[:, frame_idx] = values

                                if SHOW_PROGRESS and (frame_idx + 1) % 10 == 0:
                                        print(
                                                f"  Processed {frame_idx + 1}/{num_frames} frames",
                                                end="\r",
                                        )

                        if SHOW_PROGRESS:
                                print(
                                        f"  Processed {num_frames}/{num_frames} frames - COMPLETED"
                                )

                        # Clean up results
                        del results
                        if ENABLE_AGGRESSIVE_GC:
                                gc.collect()
                else:
                        # Sequential processing (original method or for small frame counts)
                        if not USE_PARALLEL:
                                print(
                                        "Parallel processing disabled - using sequential mode"
                                )
                        else:
                                print("Few frames detected - using sequential mode")

                        for frame_idx, (frame_num, filepath) in enumerate(frame_files):
                                if SHOW_PROGRESS:
                                        print(
                                                f"  Frame {frame_idx + 1}/{num_frames} (#{frame_num:04d}): {filepath.name}",
                                                end="",
                                                flush=True,
                                        )

                                # Read current frame
                                frame_data, _ = self.read_flood_data(filepath)

                                # Extract values for all pixels in this frame
                                if USE_VECTORIZATION:
                                        # Vectorized extraction using pre-computed arrays
                                        extracted_values = frame_data[
                                                rows_array, cols_array
                                        ]

                                        valid_mask = (extracted_values != nodata) & (
                                                extracted_values >= 0
                                        )
                                        timeseries_matrix[:, frame_idx] = NODATA_VALUE
                                        timeseries_matrix[valid_mask, frame_idx] = (
                                                extracted_values[valid_mask].astype(
                                                        np.int8
                                                )
                                        )
                                else:
                                        # Sequential extraction (original loop)
                                        for pixel_idx, (row, col, x, y) in enumerate(
                                                pixel_locations
                                        ):
                                                value = frame_data[row, col]

                                                # Store value (replace nodata with NODATA_VALUE)
                                                if value == nodata or value < 0:
                                                        timeseries_matrix[
                                                                pixel_idx, frame_idx
                                                        ] = NODATA_VALUE
                                                else:
                                                        timeseries_matrix[
                                                                pixel_idx, frame_idx
                                                        ] = int(value)

                                # CRITICAL: Close and delete frame data before next iteration
                                del frame_data
                                if (
                                        ENABLE_AGGRESSIVE_GC
                                        and (frame_idx + 1) % GC_COLLECT_FREQUENCY == 0
                                ):
                                        gc.collect()

                                if SHOW_PROGRESS:
                                        print(" : COMPLETED")

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
                        "flood_counts": np.zeros(NUM_FLOOD_CLASSES, dtype=np.int64),
                        "pixels_with_flood": 0,
                }

                # Write to CSV with chunked output for better I/O performance
                print(f"Writing to: {self.output_csv}")
                with open(self.output_csv, "w", buffering=DEFAULT_BUFFER_SIZE) as f:
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

                                        # Add frame values (replace NODATA_VALUE with empty string)
                                        for value in pixel_timeseries:
                                                if value == NODATA_VALUE:
                                                        csv_row.append(CSV_EMPTY_VALUE)
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
                                if ENABLE_AGGRESSIVE_GC:
                                        gc.collect()

                # Clear time series matrix
                del timeseries_matrix
                if ENABLE_AGGRESSIVE_GC:
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
                        for intensity in range(NUM_FLOOD_CLASSES):
                                count = stats["flood_counts"][intensity]
                                percentage = count / total_values * 100
                                label = FLOOD_INTENSITY_LABELS[intensity]
                                print(
                                        f"  {intensity} ({label:10s}): {count:10d} ({percentage:5.2f}%)"
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
                default=DEFAULT_INPUT_DIR,
                help=f"Directory containing classified flood GeoTIFF files (default: {DEFAULT_INPUT_DIR})",
        )
        parser.add_argument(
                "--output-csv",
                type=str,
                default=DEFAULT_OUTPUT_CSV,
                help=f"Output CSV file path (default: {DEFAULT_OUTPUT_CSV})",
        )
        parser.add_argument(
                "--pattern",
                type=str,
                default=DEFAULT_FILE_PATTERN,
                help=f"File pattern to match (default: {DEFAULT_FILE_PATTERN})",
        )
        parser.add_argument(
                "--sample-rate",
                type=int,
                default=DEFAULT_SAMPLE_RATE,
                help=f"Spatial sampling rate (1=all pixels, 2=every other pixel, etc.) (default: {DEFAULT_SAMPLE_RATE})",
        )
        parser.add_argument(
                "--chunk-size",
                type=int,
                default=DEFAULT_CHUNK_SIZE,
                help=f"Number of pixels to write at once (default: {DEFAULT_CHUNK_SIZE})",
        )
        parser.add_argument(
                "--valid-only",
                action="store_true",
                default=DEFAULT_VALID_ONLY,
                help="Only include pixels with valid data (not nodata)",
        )
        parser.add_argument(
                "--min-flood-events",
                type=int,
                default=DEFAULT_MIN_FLOOD_EVENTS,
                help=f"Minimum number of flood events required to include pixel (default: {DEFAULT_MIN_FLOOD_EVENTS})",
        )

        args = parser.parse_args()

        # Create extractor with configuration from arguments
        extractor = FloodTimeSeriesExtractor(
                input_dir=args.input_dir,
                output_csv=args.output_csv,
                pattern=args.pattern,
                sample_rate=args.sample_rate,
                chunk_size=args.chunk_size,
        )

        # Extract time series with frame-by-frame processing
        extractor.extract_timeseries_frame_by_frame(
                valid_pixels_only=args.valid_only,
                min_flood_events=args.min_flood_events,
        )

        print("\nTarget Extraction Complete.")


if __name__ == "__main__":
        main()
