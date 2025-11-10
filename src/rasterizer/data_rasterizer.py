import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import xy
from pathlib import Path

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

feature_files = [
        str(DATA_DIR / "static/Hanoi_DEM_30m_UTM.tif"),
        str(DATA_DIR / "static/Hanoi_HAND_30m_UTM.tif"),
        str(DATA_DIR / "static/Hanoi_imperviousness_30m_UTM.tif"),
        str(DATA_DIR / "static/Hanoi_KSAT_30m_UTM.tif"),
        str(DATA_DIR / "static/Hanoi_LULC_30m_UTM.tif"),
        str(DATA_DIR / "static/Hanoi_n_roughness_30m_UTM.tif"),
        str(DATA_DIR / "static/water_proximity_30m_UTM.tif"),
]

features = [
        "DEM",
        "HAND",
        "imperviousness",
        "KSAT",
        "LULC",
        "n_roughness",
        "water_proximity",
]

src_features = [rasterio.open(f) for f in feature_files]
feature_data = [src.read(1) for src in src_features]

h, w = feature_data[0].shape
rows, cols = np.indices((h, w))
xs, ys = xy(src_features[0].transform, rows.flatten(), cols.flatten(), offset="center")

data_dict = {"x": np.array(xs), "y": np.array(ys)}
for name, arr in zip(features, feature_data):
        data_dict[name] = arr.flatten()

df = pd.DataFrame(data_dict)

valid = np.ones(len(df), dtype=bool)
for src, name in zip(src_features, features):
        nod = src.nodata
        col = df[name].values
        if nod is not None:
                valid &= col != nod
        else:
                valid &= ~pd.isna(col)

df = df[valid]

out_csv = str(DATA_DIR / "feature_data.csv")
df.to_csv(out_csv, index=False)

for src in src_features:
        src.close()
