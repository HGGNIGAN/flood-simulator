import rasterio
import numpy as np
import os
import glob
import sys
from pathlib import Path

from src.lisflood_docker.runner import run_lisflood_docker
from src.flood_simulator.simulator import demo_simulation


# --- FILES AND PARAMETERS ---

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Absolute path to data directory
HOST_BASE_DATA_DIR = str(DATA_DIR.resolve())

# Base paths for static files
HOST_BASE_PATHS = {
        "base_data_dir": HOST_BASE_DATA_DIR,
        "dem": str(DATA_DIR / "static" / "Hanoi_DEM_30m_UTM.tif"),
        "roughness": str(DATA_DIR / "static" / "Hanoi_n_roughness_30m_UTM.tif"),
        "storm_generator_dir": str(DATA_DIR / "storm_generator"),
        "settings_dir": str(DATA_DIR / "lisflood_settings"),
        "output_dir": str(DATA_DIR / "simulation_output"),
}

# Directories inside container (recognizable by LISFLOOD docker)
CONTAINER_BASE_DATA_DIR = "/data"  # Mount for HOST_BASE_DATA_DIR

CONTAINER_BASE_PATHS = {
        "base_data_dir": CONTAINER_BASE_DATA_DIR,
        "dem": os.path.join(CONTAINER_BASE_DATA_DIR, "static", "Hanoi_DEM_30m_UTM.tif"),
        "roughness": os.path.join(
                CONTAINER_BASE_DATA_DIR, "static", "Hanoi_n_roughness_30m_UTM.tif"
        ),
        "storm_generator_dir": os.path.join(CONTAINER_BASE_DATA_DIR, "storm_generator"),
        "settings_dir": os.path.join(CONTAINER_BASE_DATA_DIR, "lisflood_settings"),
        "output_dir": os.path.join(CONTAINER_BASE_DATA_DIR, "simulation_output"),
}

# Parameters
CLASSIFICATION_BINS_METERS = [0.05, 0.25, 0.40]
CLASSIFICATION_VALUES = [0, 1, 2, 3]
# Select precipitation data type to use ("current_precip", "total_1h", "total_2h", "total_24h")
PRECIP_TYPE = "total_2h"

# --- HELPER FUNCTIONS ---


def get_storm_frames(precip_type="current_precip"):
        storm_dir = os.path.join(HOST_BASE_PATHS["storm_generator_dir"], precip_type)

        if not os.path.exists(storm_dir):
                print(f"Warning: Directory {storm_dir} does not exist!")
                return []

        # Find all .tif files in directory
        pattern = os.path.join(storm_dir, "total_*.tif")
        files = sorted(glob.glob(pattern))

        print(f"Found {len(files)} frames in {precip_type}")
        return files


def get_frame_number(filepath):
        basename = os.path.basename(filepath)
        # Format: precip_frame_XXXX.tif
        frame_str = basename.split("_")[-1].replace(".tif", "")
        return int(frame_str)


def reclassify_flood_depth(raw_depth_file, classified_file, bins, nodata_val=-9999.0):
        print(
                f"... Starting 4-level classification (Step 3) from file: {raw_depth_file}"
        )

        with rasterio.open(raw_depth_file) as src:
                raw_depth_meters = src.read(1)
                profile = src.profile

                valid_mask = raw_depth_meters > nodata_val
                classified_data = np.full(raw_depth_meters.shape, -1, dtype=np.int8)

                classified_data[valid_mask] = np.digitize(
                        raw_depth_meters[valid_mask], bins=bins
                ).astype(np.int8)

                # Update profile for output file
                profile.update(dtype="int8", count=1, compress="lzw", nodata=-1)

                with rasterio.open(classified_file, "w", **profile) as dst:
                        dst.write(classified_data, 1)

        print(f"Saved classification result: {classified_file}")


# --- STORM GENERATION FUNCTIONS ---


def run_storm_generator(use_test_version=False):
        """Run storm generator (full or test version)"""
        print("\n" + "=" * 80)
        if use_test_version:
                print("RUNNING TEST STORM GENERATOR")
                print("Output directory: data/storm_generator_test/")
        else:
                print("RUNNING FULL STORM GENERATOR")
                print("Output directory: data/storm_generator/")
        print("=" * 80)

        try:
                if use_test_version:
                        # Import and run test version
                        sys.path.insert(0, str(PROJECT_ROOT / "src"))
                        from storm_gen import storm_generator_test

                        storm_generator_test.run_simulation()
                else:
                        # Import and run full version
                        sys.path.insert(0, str(PROJECT_ROOT / "src"))
                        from storm_gen import storm_generator

                        storm_generator.run_simulation()

                print("\nDONE: Storm generation completed successfully!")
                return True
        except Exception as e:
                print(f"\nERROR: Storm generation failed: {e}")
                import traceback

                traceback.print_exc()
                return False


def run_precipitation_rasterizer(use_test_version=False):
        """Extract precipitation data to CSV"""
        print("\n" + "=" * 80)
        print("EXTRACTING PRECIPITATION DATA TO CSV")
        print("=" * 80)

        try:
                sys.path.insert(0, str(PROJECT_ROOT / "src"))
                from rasterizer import data_storm_rasterizer

                # Run with test flag if needed
                data_storm_rasterizer.main(use_test_output=use_test_version)

                print("\nDONE: Precipitation extraction completed successfully!")
                return True
        except Exception as e:
                print(f"\nERROR: Precipitation extraction failed: {e}")
                import traceback

                traceback.print_exc()
                return False


def run_flood_rasterizer():
        """Extract flood classification data to CSV"""
        print("\n" + "=" * 80)
        print("EXTRACTING FLOOD CLASSIFICATION DATA TO CSV")
        print("=" * 80)

        try:
                sys.path.insert(0, str(PROJECT_ROOT / "src"))
                from rasterizer import data_target_rasterizer

                extractor = data_target_rasterizer.FloodTimeSeriesExtractor(
                        sample_rate=1,  # Use all pixels
                        chunk_size=80000,
                )

                # Run extraction
                extractor.extract_timeseries_frame_by_frame(
                        valid_pixels_only=True, min_flood_events=0
                )

                print("\nDONE: Flood classification extraction completed successfully!")
                return True
        except Exception as e:
                print(f"\nERROR: Flood classification extraction failed: {e}")
                import traceback

                traceback.print_exc()
                return False


# --- MAIN PROGRAM ---


def main():
        print("=" * 80)
        print("FLOOD SIMULATION PIPELINE")
        print("=" * 80)

        # Step 0: Optional storm generation
        print("\n" + "-" * 80)
        print("STEP 0: STORM GENERATION (OPTIONAL)")
        print("-" * 80)

        use_test_storm = False

        response = (
                input("\nDo you want to generate new storm data? (y/n): ")
                .strip()
                .lower()
        )

        if response == "y":
                # Ask which version to use
                response_test = (
                        input("Use test storm generator (faster, simpler)? (y/n): ")
                        .strip()
                        .lower()
                )
                use_test_storm = response_test == "y"

                # Run storm generator
                success = run_storm_generator(use_test_version=use_test_storm)
                if success:
                        # Ask if user wants to extract precipitation CSV
                        response_csv = (
                                input("\nExtract precipitation data to CSV? (y/n): ")
                                .strip()
                                .lower()
                        )
                        if response_csv == "y":
                                run_precipitation_rasterizer(
                                        use_test_version=use_test_storm
                                )
                else:
                        print(
                                "\nERROR: Storm generation failed. Using existing storm data..."
                        )
        else:
                print("Skipping storm generation. Using existing storm data...")

        print("\n" + "=" * 80)
        print("STEP 1: FLOOD SIMULATION")
        print("Processing all frames from Storm Generator")
        print("=" * 80)

        # Ensure output directories exist
        os.makedirs(HOST_BASE_PATHS["output_dir"], exist_ok=True)
        os.makedirs(HOST_BASE_PATHS["settings_dir"], exist_ok=True)

        # Get all frames from storm generator
        storm_frames = get_storm_frames(precip_type=PRECIP_TYPE)

        if not storm_frames:
                print(
                        f"No files found in {PRECIP_TYPE}. Please run storm_generator.py first."
                )
                return

        print(
                f"\nWill process {len(storm_frames)} frames from '{PRECIP_TYPE}' directory"
        )
        print("-" * 80)

        # Run mode
        # IMPORTANT: Extensive configuration is needed for LISFLOOD to run properly
        USE_DOCKER = False  # True: run with Docker, False: run demo simulation
        LISFLOOD_IMAGE = "jrce1/lisflood:latest"

        # Statistics
        successful_frames = 0
        failed_frames = 0

        # Loop through all frames
        for idx, rain_file in enumerate(storm_frames, 1):
                frame_num = get_frame_number(rain_file)
                print(
                        f"\n[{idx}/{len(storm_frames)}] Processing Frame {frame_num:04d}..."
                )
                print(f"    Input: {os.path.basename(rain_file)}")

                # Create paths for this frame
                HOST_PATHS = {
                        "base_data_dir": HOST_BASE_PATHS["base_data_dir"],
                        "dem": HOST_BASE_PATHS["dem"],
                        "roughness": HOST_BASE_PATHS["roughness"],
                        "rain": rain_file,
                        "settings": os.path.join(
                                HOST_BASE_PATHS["settings_dir"],
                                f"settings_frame_{frame_num:04d}.xml",
                        ),
                        "output_raw": os.path.join(
                                HOST_BASE_PATHS["output_dir"],
                                f"flood_depth_raw_{frame_num:04d}.tif",
                        ),
                        "output_classified": os.path.join(
                                HOST_BASE_PATHS["output_dir"],
                                f"flood_classified_{frame_num:04d}.tif",
                        ),
                }

                CONTAINER_PATHS = {
                        "base_data_dir": CONTAINER_BASE_PATHS["base_data_dir"],
                        "dem": CONTAINER_BASE_PATHS["dem"],
                        "roughness": CONTAINER_BASE_PATHS["roughness"],
                        "rain": os.path.join(
                                CONTAINER_BASE_PATHS["storm_generator_dir"],
                                PRECIP_TYPE,
                                os.path.basename(rain_file),
                        ),
                        "settings": os.path.join(
                                CONTAINER_BASE_PATHS["settings_dir"],
                                f"settings_frame_{frame_num:04d}.xml",
                        ),
                        "output": os.path.join(
                                CONTAINER_BASE_PATHS["output_dir"],
                                f"flood_depth_raw_{frame_num:04d}.tif",
                        ),
                }

                # Run hydraulic model
                if USE_DOCKER:
                        print("    Running LISFLOOD with Docker...")
                        success = run_lisflood_docker(
                                HOST_PATHS, CONTAINER_PATHS, LISFLOOD_IMAGE
                        )
                else:
                        print("    [DEMO MODE] Creating simulation data...")
                        success = demo_simulation(HOST_PATHS)

                if not success:
                        print(f"    ❌ Frame {frame_num:04d} failed!")
                        failed_frames += 1
                        continue

                # 4-Level Classification
                try:
                        reclassify_flood_depth(
                                raw_depth_file=HOST_PATHS["output_raw"],
                                classified_file=HOST_PATHS["output_classified"],
                                bins=CLASSIFICATION_BINS_METERS,
                        )
                        print(f"    DONE: Frame {frame_num:04d} completed!")
                        successful_frames += 1
                except Exception as e:
                        print(f"    ERROR: Frame {frame_num:04d}: {e}")
                        failed_frames += 1

        # Summary
        print("\n" + "=" * 80)
        print("STEP 1 COMPLETED: FLOOD SIMULATION")
        print("=" * 80)
        print(f"Total frames: {len(storm_frames)}")
        print(f"Successful: {successful_frames}")
        print(f"Failed: {failed_frames}")
        print(f"\nResults saved to: {HOST_BASE_PATHS['output_dir']}")
        print("  - Raw flood depth: flood_depth_raw_XXXX.tif")
        print("  - Classified: flood_classified_XXXX.tif")

        # Step 2: Optional flood data extraction
        if successful_frames > 0:
                print("\n" + "-" * 80)
                print("STEP 2: FLOOD DATA EXTRACTION (OPTIONAL)")
                print("-" * 80)

                response_flood_csv = (
                        input("\nExtract flood classification data to CSV? (y/n): ")
                        .strip()
                        .lower()
                )

                if response_flood_csv == "y":
                        run_flood_rasterizer()

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED")
        print("=" * 80)


if __name__ == "__main__":
        main()
