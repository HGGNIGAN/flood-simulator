import rasterio
import numpy as np
import polars as pl
from rasterio.transform import xy
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Define the storm data directories for each time aggregation
STORM_DIRS = {
        # "current": DATA_DIR / "storm_generator" / "current_precip",
        "1h": DATA_DIR / "storm_generator" / "total_1h",
        "2h": DATA_DIR / "storm_generator" / "total_2h",
        "24h": DATA_DIR / "storm_generator" / "total_24h",
}


def process_precipitation_files(time_period: str, input_dir: Path, output_csv: Path):
        print(f"\nProcessing {time_period} precipitation data...")

        # Find all precipitation frame files
        precip_files = sorted(input_dir.glob("total_*.tif"))

        if not precip_files:
                print(f"ERROR:  No precipitation files found in {input_dir}")
                return

        print(f"  Found {len(precip_files)} precipitation frames")

        # We'll use the first file to establish reference grid and coordinates
        xs = None
        ys = None
        ref_shape = None

        # Collect all precipitation data frames
        all_precip_data = []
        frame_numbers = []

        for i, precip_file in enumerate(
                tqdm(precip_files, desc=f"  Reading {time_period} frames")
        ):
                with rasterio.open(precip_file) as src:
                        # Validate alignment with first raster
                        if i == 0:
                                ref_shape = (src.height, src.width)

                                # Generate coordinates from the first file (same as data_rasterizer.py)
                                h, w = ref_shape
                                rows, cols = np.indices((h, w))
                                xs, ys = xy(
                                        src.transform,
                                        rows.flatten(),
                                        cols.flatten(),
                                        offset="center",
                                )
                                xs = np.array(xs)
                                ys = np.array(ys)

                        # Read as masked array to handle nodata robustly
                        precip_data = src.read(1, masked=True)

                        # Extract frame number from filename (e.g., precip_frame_0042.tif -> 42)
                        frame_num = int(precip_file.stem.split("_")[-1])
                        frame_numbers.append(frame_num)

                        # Store the precipitation data
                        all_precip_data.append(precip_data)

        print(f"  Creating DataFrame with {len(all_precip_data)} frames...")

        # Build the dataframe with coordinates and all precipitation frames
        # Using polars for better performance with large datasets
        data_dict = {
                "x": xs,
                "y": ys,
        }

        # Add each precipitation frame as a column
        for frame_num, precip_arr in zip(frame_numbers, all_precip_data):
                col_name = f"precip_frame_{frame_num:04d}"
                data_dict[col_name] = precip_arr.data.flatten()

        df = pl.DataFrame(data_dict)

        # Filter out invalid pixels (masked in ANY frame)
        print("  Filtering invalid pixels...")
        valid = np.ones(len(df), dtype=bool)
        for precip_arr in all_precip_data:
                valid &= ~precip_arr.mask.flatten()

        df = df.filter(pl.Series(valid))

        print(f"  Valid pixels: {len(df):,}")

        # Save to CSV
        print(f"  Saving to {output_csv.name}...")
        df.write_csv(str(output_csv))

        print(f"SUCCESS: {time_period} precipitation data saved to {output_csv}")


def main():
        for time_period, input_dir in STORM_DIRS.items():
                output_csv = DATA_DIR / f"precipitation_{time_period}.csv"
                try:
                        process_precipitation_files(time_period, input_dir, output_csv)
                except Exception as e:
                        print(f"Failed to process {time_period}: {e}")
                        continue

        print()
        print("Processed all precipitation data.\n")


if __name__ == "__main__":
        main()
