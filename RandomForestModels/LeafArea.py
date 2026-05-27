# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:33:40 2026

@author: cenv1124
"""

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
# Load data
# -----------------------------
Amax = pd.read_csv(
    "R:/GlobalDataset/TraitsCombinedWithGeoTessera/FinalizedTesseraData/LeafAreaCombined.csv"
)
Amax = Amax[Amax['label'].isin([1, 3, 5])]  # trees, flooded vegetation, shrubs

# Initial drop of missing target or missing raw embedding rows
df = Amax.dropna(subset=["0", "Leaf area (cm2)"]).copy()

# -----------------------------
# Handle Categorical Land Cover Labels
# -----------------------------
# One-hot encode the label column so RF doesn't treat 1, 3, 5 as ordered quantities
df = pd.get_dummies(df, columns=['label'], prefix='type', drop_first=False)
type_cols = [c for c in df.columns if c.startswith('type_')]

# Base environmental/soil columns
soil_base_cols = [
    "CEC",
    "Clay",
    "Sand",
    "pH",
    "slope",
    "mcwd",
    "tmax_mean"
]
soil_cols = soil_base_cols + type_cols

# -----------------------------
# Isolate GeoTessera embedding columns
# -----------------------------
geo_cols = [c for c in df.columns if c.isdigit()]

# Remove low-variance dimensions
geo_cols = [c for c in geo_cols if df[c].std() > 0.01]

# -----------------------------
# NON-LINEAR FEATURE SELECTION
# -----------------------------
print("Running non-linear feature selection on GeoTessera dimensions...")

# Use a fast, shallow forest to score which spatial dimensions actually reduce error
selector_rf = RandomForestRegressor(
    n_estimators=1000, 
    max_depth=15, 
    random_state=42, 
    n_jobs=-1
)
selector_rf.fit(df[geo_cols], df["Leaf area (cm2)"])

# Rank them by feature importance and keep the top 15 dimensions
importances = pd.Series(selector_rf.feature_importances_, index=geo_cols)
geo_cols = importances.sort_values(ascending=False).head(20).index.tolist()

print("\nTop 15 Non-Linear GeoTessera dimensions selected:")
print(geo_cols)

# -----------------------------
# Final Predictor Columns & Data Cleaning
# -----------------------------
predictor_cols = geo_cols + soil_cols

# Drop any rows missing your chosen soil/climate predictors all at once
df = df.dropna(subset=predictor_cols)

# -----------------------------
# Feature matrix & Target
# -----------------------------
X = df[predictor_cols]
y = df["Leaf area (cm2)"]

# -----------------------------
# Spatial blocking
# -----------------------------
lat_col = "Coords_y"
lon_col = "Coords_x"
grid_size = 1  # ~1 degree grid blocks

df["lat_bin"] = df[lat_col] // grid_size
df["lon_bin"] = df[lon_col] // grid_size

df["spatial_block"] = (
    df["lat_bin"].astype(str)
    + "_"
    + df["lon_bin"].astype(str)
)

groups = df["spatial_block"]

print("\nTotal spatial blocks:", df["spatial_block"].nunique())

# -----------------------------
# Main Random Forest Model Configuration
# -----------------------------
model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=15,          # Controls spatial overfitting
    min_samples_leaf=15,       # Forces broader global generalization
    min_samples_split=30,      
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# Spatial CV Evaluation
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
# Print Results
# -----------------------------
print("\n===== Spatial CV Results =====")
print("Mean Train R2:", round(scores["train_r2"].mean(), 3))
print("Mean Test R2:", round(scores["test_r2"].mean(), 3))
print("Mean RMSE:", round(-scores["test_rmse"].mean(), 3))
print("Mean MAE:", round(-scores["test_mae"].mean(), 3))

# -----------------------------
# Final training
# -----------------------------
print("\nTraining final model...")
model.fit(X, y)

print("\nDone ✅")



