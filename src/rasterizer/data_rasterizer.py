import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import xy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

feature_files = [
        str(DATA_DIR / "static/Hanoi_DEM_30m_UTM.tif"),
        str(DATA_DIR / "static/Hanoi_Slope_30m_UTM.tif"),
        str(DATA_DIR / "static/Hanoi_HAND_30m_UTM.tif"),
        str(DATA_DIR / "static/Hanoi_imperviousness_30m_UTM.tif"),
        str(DATA_DIR / "static/Hanoi_KSAT_30m_UTM.tif"),
        str(DATA_DIR / "static/Hanoi_LULC_30m_UTM.tif"),
        str(DATA_DIR / "static/Hanoi_n_roughness_30m_UTM.tif"),
        str(DATA_DIR / "static/water_proximity_30m_UTM.tif"),
]

features = [
        "DEM",
        "Slope",
        "HAND",
        "imperviousness",
        "KSAT",
        "LULC",
        "n_roughness",
        "water_proximity",
]

src_features = []
feature_data = []


def main():
        for i, f in enumerate(feature_files):
                src = rasterio.open(f)
                src_features.append(src)

        # Read as masked arrays to handle nodata robustly
        # Masked arrays automatically handle nodata values including NaN
        feature_data = [src.read(1, masked=True) for src in src_features]

        h, w = feature_data[0].shape
        rows, cols = np.indices((h, w))
        xs, ys = xy(
                src_features[0].transform,
                rows.flatten(),
                cols.flatten(),
                offset="center",
        )

        data_dict = {"x": np.array(xs), "y": np.array(ys)}
        for name, arr in zip(features, feature_data):
                data_dict[name] = arr.data.flatten()

        df = pd.DataFrame(data_dict)

        # Use mask from masked arrays instead of manual nodata checking
        # This handles NaN nodata, missing nodata metadata, and all edge cases
        valid = np.ones(len(df), dtype=bool)
        for arr in feature_data:
                # Combine masks from all rasters - exclude any pixel that's masked in any layer
                valid &= ~arr.mask.flatten()

        df = df[valid]

        out_csv = str(DATA_DIR / "feature_data.csv")
        df.to_csv(out_csv, index=False)

        print(f"Feature data saved to {out_csv}")
        for src in src_features:
                src.close()


if __name__ == "__main__":
        main()
