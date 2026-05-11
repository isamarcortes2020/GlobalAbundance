# -*- coding: utf-8 -*-
"""
Spatial cross-validation + GeoTessera prediction + yearly mosaics
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate

# -----------------------------
# Load data
# -----------------------------
Amax = pd.read_csv(
    "R:/GlobalDataset/TraitsCombinedWithGeoTessera/FinalizedTesseraData/LeafHydraulicCombined.csv"
)

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
    "tmax_mean"
]

# GeoTessera embedding columns
geo_cols = [c for c in df.columns if c.isdigit()]


# remove low-variance dimensions
geo_cols = [
    c for c in geo_cols
    if df[c].std() > 0.01
]

# -----------------------------
# Select top embedding dimensions
# -----------------------------
corrs = (
    df[geo_cols]
    .corrwith(df["Leaf hydraulic conductance mmol m-2 s-1 MPa-1"])
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
y = df["Leaf hydraulic conductance mmol m-2 s-1 MPa-1"]

# optional:
# y = np.log1p(df["Asat"])

# -----------------------------
# Spatial blocking
# -----------------------------
lat_col = "Coords_y"
lon_col = "Coords_x"

# ~50 km blocks
grid_size = 0.5

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

# -----------------------------
# Feature importance
# -----------------------------
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)



print("\nDone ✅")
