# -*- coding: utf-8 -*-
"""
Spatial cross-validation + GeoTessera prediction + yearly mosaics
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.merge import merge
from geotessera import GeoTessera

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate

# -----------------------------
# Load data
# -----------------------------
Amax = pd.read_csv(
    'R:/GlobalDataset/TraitsCombinedWithGeoTessera/UpdatedDatasetSoFar/LeafCaContent/LeafCaContent2017_Present.csv'
)

df = Amax.dropna(subset=["0"]).copy()

# -----------------------------
# Feature definition (FIXED)
# -----------------------------
soil_cols = ["CEC", "Clay", "Sand", "pH"]
geo_cols = [c for c in df.columns if c.isdigit()]
geo_dim = len(geo_cols)

X = df[geo_cols + soil_cols]
y = df["Leaf Ca content (%)"]

# -----------------------------
# Column names
# -----------------------------
lat_col = "Coords_y"
lon_col = "Coords_x"
year_col = "Year"

# -----------------------------
# Spatial blocking
# -----------------------------
grid_size = 1.0

df["lat_bin"] = (df[lat_col] // grid_size)
df["lon_bin"] = (df[lon_col] // grid_size)

df["spatial_block"] = (
    df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
)

groups = df["spatial_block"]

print("\nSpatial block sizes (top 10):")
print(df["spatial_block"].value_counts().head(10))
print("\nTotal number of spatial blocks:", df["spatial_block"].nunique())

# -----------------------------
# Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# Spatial CV
# -----------------------------
gkf = GroupKFold(n_splits=2)

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
    return_train_score=False
)

print("\n===== Spatial Cross-Validation Results =====")
print("Mean R2:", scores["test_r2"].mean())
print("Mean RMSE:", -scores["test_rmse"].mean())
print("Mean MAE:", -scores["test_mae"].mean())

# -----------------------------
# Train final model
# -----------------------------
print("\nTraining final model...")
model.fit(X, y)

# -----------------------------
# GeoTessera setup
# -----------------------------
gt = GeoTessera()

out_dir = "R:/GlobalDataset/GeoTesseraOutputs/LeafCa_100m/"
os.makedirs(out_dir, exist_ok=True)

factor = 10  # 10m → 100m

# FIX: keep soil WITH coordinates
unique_coords = df[[lon_col, lat_col, year_col] + soil_cols].drop_duplicates().reset_index(drop=True)

print("\nGenerating tiles...")

chunk_size = 50000

for i, row in unique_coords.iterrows():
    lon = row[lon_col]
    lat = row[lat_col]
    year = int(row[year_col])

    out_name = os.path.join(
        out_dir,
        f"leafca_{lat:.3f}_{lon:.3f}_{year}.tif"
    )

    if os.path.exists(out_name):
        continue

    try:
        # Fetch embedding tile
        tile_data, crs, transform = gt.fetch_embedding(
            lon=lon,
            lat=lat,
            year=year
        )

        # -----------------------------
        # Aggregate to 100 m
        # -----------------------------
        h, w, c = tile_data.shape

        h_new = h // factor
        w_new = w // factor

        tile_data = tile_data[:h_new*factor, :w_new*factor]

        tile_data_coarse = tile_data.reshape(
            h_new, factor,
            w_new, factor,
            c
        ).mean(axis=(1, 3))

        # -----------------------------
        # Build prediction features (FIXED)
        # -----------------------------
        geo_pixels = tile_data_coarse.reshape(-1, geo_dim)

        # FIX: safe soil extraction
        soil_values = row.reindex(soil_cols).fillna(0).values.astype(float)
        soil_pixels = np.tile(soil_values, (geo_pixels.shape[0], 1))

        pixels = np.hstack([geo_pixels, soil_pixels])

        # -----------------------------
        # Predict (chunked)
        # -----------------------------
        preds_list = []
        for j in range(0, len(pixels), chunk_size):
            chunk = pixels[j:j + chunk_size]
            preds_list.append(model.predict(chunk))

        preds = np.concatenate(preds_list)

        trait_map = preds.reshape(h_new, w_new)

        # -----------------------------
        # Save GeoTIFF
        # -----------------------------
        new_transform = transform * Affine.scale(factor, factor)

        with rasterio.open(
            out_name,
            "w",
            driver="GTiff",
            height=h_new,
            width=w_new,
            count=1,
            dtype=trait_map.dtype,
            crs=crs,
            transform=new_transform,
        ) as dst:
            dst.write(trait_map, 1)

        print(f"Saved {out_name}")

    except Exception as e:
        print(f"Failed at {lat}, {lon}, {year}: {e}")

# -----------------------------
# Merge tiles into mosaics
# -----------------------------
print("\nMerging yearly mosaics...")

years = unique_coords[year_col].unique()

for year in years:
    print(f"\nProcessing year {year}")

    tif_files = glob.glob(os.path.join(out_dir, f"*_{int(year)}.tif"))

    if len(tif_files) == 0:
        print("No tiles found")
        continue

    src_files = [rasterio.open(fp) for fp in tif_files]

    mosaic, out_transform = merge(src_files)

    out_path = os.path.join(out_dir, f"leafca_mosaic_{int(year)}.tif")

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=1,
        dtype=mosaic.dtype,
        crs=src_files[0].crs,
        transform=out_transform,
    ) as dst:
        dst.write(mosaic[0], 1)

    print(f"Saved mosaic: {out_path}")

    for src in src_files:
        src.close()

print("\n✅ DONE")
