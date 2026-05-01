# -*- coding: utf-8 -*-
"""
Spatial cross-validation pipeline (grid-based blocking)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate

# -----------------------------
# Load data
# -----------------------------
Amax = pd.read_csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/Amax.csv")

# Drop missing embeddings (SAFE copy to avoid warning)
df = Amax.dropna(subset=["0"]).copy()
#df = df.dropna(subset=["Vessel lumen area (mm2)"]).copy()
# -----------------------------
# Features and target
# -----------------------------
embeddings_only = df.loc[:, "0":]

X = embeddings_only
y = df["Amax"]

# -----------------------------
# ⚠️ REQUIREMENT: coordinates
# -----------------------------
# Change these if your column names differ
lat_col = "Coords_y"
lon_col = "Coords_x"

# -----------------------------
# Grid-based spatial blocking
# -----------------------------
grid_size = 5.0  # degrees (try 1–5 depending on dataset)

df["lat_bin"] = (df[lat_col] // grid_size)
df["lon_bin"] = (df[lon_col] // grid_size)

df["spatial_block"] = (
    df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
)

groups = df["spatial_block"]

# Check block sizes
print("\nSpatial block sizes (top 10):")
print(df["spatial_block"].value_counts().head(10))

print("\nTotal number of spatial blocks:", df["spatial_block"].nunique())

# -----------------------------
# Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# Spatial cross-validation
# -----------------------------
n_splits = 5  # reduce to 3 if needed

gkf = GroupKFold(n_splits=n_splits)

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

# -----------------------------
# Results
# -----------------------------
r2_scores = scores["test_r2"]
rmse_scores = -scores["test_rmse"]
mae_scores = -scores["test_mae"]

print("\n===== Spatial Cross-Validation Results =====")
print("R2 per fold:", r2_scores)
print("Mean R2:", r2_scores.mean())
print("Std R2:", r2_scores.std())

print("\nRMSE per fold:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())

print("\nMAE per fold:", mae_scores)
print("Mean MAE:", mae_scores.mean())
