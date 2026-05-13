# -*- coding: utf-8 -*-
"""
Spatial cross-validation + GeoTessera prediction + yearly mosaics
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate
import os
import glob

# -----------------------------

# -----------------------------
# Load data
# -----------------------------
Amax = pd.read_csv(
    "R:/GlobalDataset/TraitsCombinedWithGeoTessera/FinalizedTesseraData/LeafCaContentCombined.csv"
)
Amax = Amax[Amax['label'].between(1, 5)]
# remove missing target + embeddings
df = Amax.dropna(subset=["0"]).copy()

# -----------------------------
# Predictor columns
# -----------------------------
soil_cols = [
    "CEC",
    "Clay",
    "Sand",
    "pH",
    "slope",
    "mcwd",
    "tmax_mean",
    "Coords_x",
    "Coords_y",
    "label"
]

# GeoTessera embedding columns
geo_cols = [c for c in df.columns if c.isdigit()]


# remove low-variance dimensions

geo_cols = [
    c for c in geo_cols
    if df[c].std() > 0.01
]
'''
# -----------------------------
# Select top embedding dimensions
# -----------------------------
corrs = (
    df[geo_cols]
    .corrwith(df["Leaf Ca content (%)"])
    .abs()
)

geo_cols = (
    corrs
    .sort_values(ascending=False)
    .head(20)
    .index
    .tolist()
)

print("\nTop GeoTessera dimensions:")
print(geo_cols)
'''


# -----------------------------
# Remove missing predictor rows
# -----------------------------
predictor_cols = (
    geo_cols
    + soil_cols
)

df = df.dropna(subset=predictor_cols)

# -----------------------------
# Feature matrix
# -----------------------------
X = df[predictor_cols]



# target
y = df["Leaf Ca content (%)"]

# optional:
# y = np.log1p(df["Asat"])

# -----------------------------
# Spatial blocking
# -----------------------------
lat_col = "Coords_y"
lon_col = "Coords_x"
year_col = "Year"
# ~50 km blocks
grid_size = 1

df["lat_bin"] = (
    df[lat_col] // grid_size
)

df["lon_bin"] = (
    df[lon_col] // grid_size
)

df["spatial_block"] = (
    df["lat_bin"].astype(str)
    + "_"
    + df["lon_bin"].astype(str)
)

groups = df["spatial_block"]

print(
    "\nTotal spatial blocks:",
    df["spatial_block"].nunique()
)

# -----------------------------
# Random Forest
# -----------------------------
model = RandomForestRegressor(
    n_estimators=1500,
    max_depth=15,
    min_samples_leaf=10,
    min_samples_split=10,
    max_features=0.3,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
# -----------------------------
# Spatial CV
# -----------------------------
gkf = GroupKFold(n_splits=5)

scores = cross_validate(
    model,
    X,
    y,
    cv=gkf,
    groups=groups,
    scoring={
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error"
    },
    n_jobs=-1,
    return_train_score=True
)

# -----------------------------
# Results
# -----------------------------
print("\n===== Spatial CV Results =====")

print(
    "Mean Train R2:",
    round(scores["train_r2"].mean(), 3)
)

print(
    "Mean Test R2:",
    round(scores["test_r2"].mean(), 3)
)

print(
    "Mean RMSE:",
    round(-scores["test_rmse"].mean(), 3)
)

print(
    "Mean MAE:",
    round(-scores["test_mae"].mean(), 3)
)

# -----------------------------
# Final training
# -----------------------------
print("\nTraining final model...")

model.fit(X, y)


print("\nDone ✅")

import os
import numpy as np
import rasterio
from affine import Affine
from geotessera import GeoTessera

# -----------------------------
# setup
# -----------------------------
gt = GeoTessera()
out_dir = "R:/GlobalDataset/GeoTesseraOutputs/LeafCa_100m"
os.makedirs(out_dir, exist_ok=True)

chunk_size = 50000  # prevents memory spikes

# df must contain: lon_col, lat_col, year_col
coords = df[[lon_col, lat_col, year_col]].drop_duplicates().reset_index(drop=True)

coords[year_col] = coords[year_col].replace(
    {y: 2017 for y in range(2010, 2017)}
)


# -----------------------------
# loop over space-time points
# -----------------------------
for row in coords.itertuples(index=False):

    lon = float(getattr(row, lon_col))
    lat = float(getattr(row, lat_col))
    year = int(getattr(row, year_col))

    out_path = os.path.join(
        out_dir,
        f"trait_{lat:.3f}_{lon:.3f}_{year}.tif"
    )

    if os.path.exists(out_path):
        continue

    try:
        # -----------------------------
        # 1. fetch tessera embedding
        # -----------------------------
        tile_data, crs, transform = gt.fetch_embedding(
            lon=lon,
            lat=lat,
            year=year
        )

        h, w, c = tile_data.shape

        # -----------------------------
        # 2. flatten embedding
        # -----------------------------
        geo_pixels = tile_data.reshape(-1, c)

        # -----------------------------
        # 3. get soil for this location
        # (safe nearest-match approach)
        # -----------------------------
        idx = ((df[lon_col] - lon)**2 + (df[lat_col] - lat)**2).idxmin()

        soil_values = df.loc[idx, soil_cols].values.astype(float)

        soil_pixels = np.tile(
            soil_values,
            (geo_pixels.shape[0], 1)
        )

        # -----------------------------
        # 4. combine features (CRITICAL)
        # -----------------------------
        pixels = np.hstack([geo_pixels, soil_pixels])

        # -----------------------------
        # 5. predict in chunks
        # -----------------------------
        preds = []

        for i in range(0, len(pixels), chunk_size):
            chunk = pixels[i:i + chunk_size]
            preds.append(model.predict(chunk))

        preds = np.concatenate(preds)

        # -----------------------------
        # 6. reshape back to raster
        # -----------------------------
        # reshape to original raster
        trait_map = preds.reshape(h, w).astype(np.float32)
        
        # aggregate 10m -> 100m
        factor = 10
        
        # trim edges to divisible dimensions
        h_trim = (h // factor) * factor
        w_trim = (w // factor) * factor
        
        trait_map = trait_map[:h_trim, :w_trim]
        
        # block mean aggregation
        trait_map = trait_map.reshape(
            h_trim // factor,
            factor,
            w_trim // factor,
            factor
        ).mean(axis=(1, 3)).astype(np.float32)


        transform = Affine(
            transform.a * factor,
            transform.b,
            transform.c,
            transform.d,
            transform.e * factor,
            transform.f
        )
        h, w = trait_map.shape
        # -----------------------------
        # 7. save GeoTIFF
        # -----------------------------
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(trait_map, 1)

        print(f"Saved {out_path}")

    except Exception as e:
        print(f"Failed at {lat}, {lon}, {year}: {e}")
        
