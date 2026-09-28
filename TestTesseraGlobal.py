# -*- coding: utf-8 -*-
"""
Lazy TESSERA -> Random Forest -> forest-only GeoTIFF workflow.

Key changes from the original workflow:
- No ProcessPoolExecutor / multiprocessing.
- Training rows are still filtered to label 1, 3, 5.
- The CSV label is NOT converted into type_* predictors.
- Predictors are selected GeoTessera dimensions + continuous environmental rasters.
- The forest GeoTIFF defines the prediction area and is a separate mask:
      0 = forest -> retain prediction
      1 = non-forest -> NoData
- GeoTessera is streamed lazily with iter_region().
- Predictions are written directly into a fixed EPSG:6933 output grid.

Change the paths in CONFIGURATION before running.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin, array_bounds
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject, transform_bounds

from geotessera import GeoTesseraZarr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAINING_CSV = (
    "R:/GlobalDataset/TraitsCombinedWithGeoTessera/FinalizedTesseraData/AmaxCombined.csv"
)

FOREST_RASTER = "C:/Users/cenv1124/Downloads/foresttest.tif"

OUTPUT_RASTER = (
    "C:/Users/cenv1124/Downloads/Amax_2024_forest_only_EPSG6933.tif"
)

TARGET_CRS = "EPSG:6933"
YEAR = 2024

# Forest raster semantics supplied by you.
FOREST_VALUE = 0

# TESSERA is 10 m; original workflow aggregated by factor 10.
TESSERA_RESOLUTION = 10.0
AGGREGATION_FACTOR = 10
OUTPUT_RESOLUTION = (
    TESSERA_RESOLUTION * AGGREGATION_FACTOR
)

# Number of TESSERA rows materialized at a time.
STRIP_ROWS = 128

# Number of selected GeoTessera dimensions.
N_GEO_FEATURES = 20

SOIL_RASTER_PATHS = {
    "CEC": (
        "R:/GlobalDataset/EnvironmentalData/cec_avg.tif"
    ),
    "Clay": (
        "R:/GlobalDataset/EnvironmentalData/clay_avg.tif"
    ),
    "Sand": (
        "R:/GlobalDataset/EnvironmentalData/sand_avg.tif"
    ),
    "pH": (
        "R:/GlobalDataset/EnvironmentalData/pH_avg.tif"
    ),
    "slope": (
        "R:/GlobalDataset/EnvironmentalData/SRTM_global.tif"
    ),
    "mcwd_mean": (
        "R:/GlobalDataset/EnvironmentalData/MCWD_mean_1988_2017.tif"
    ),
    "tmax_mean": (
        "R:/GlobalDataset/EnvironmentalData/Tmax_mean_1988_2017.tif"
    ),
}


# =============================================================================
# VALIDATION
# =============================================================================

def validate_paths():
    paths = {
        "Training CSV": TRAINING_CSV,
        "Forest raster": FOREST_RASTER,
    }
    paths.update(
        {
            f"Soil raster: {name}": path
            for name, path in SOIL_RASTER_PATHS.items()
        }
    )

    for label, path in paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{label} does not exist:\n{path}"
            )
        print(f"OK: {label}")

    Path(OUTPUT_RASTER).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_model():
    """
    Train the model.

    `label` is used ONLY to retain training rows 1, 3 and 5.
    It is not converted into type_* columns and is not a predictor.
    """

    print("\n" + "=" * 72)
    print("STEP 1 - TRAIN MODEL")
    print("=" * 72)

    df = pd.read_csv(TRAINING_CSV)
    print(f"Original rows: {len(df):,}")

    df = df[df["label"].isin([1, 3, 5])].copy()
    print(f"Rows after label filter [1, 3, 5]: {len(df):,}")

    df = df.dropna(
        subset=["0", "Amax"]
    ).copy()

    geo_cols = [
        c for c in df.columns
        if str(c).isdigit()
    ]
    geo_cols = [
        c for c in geo_cols
        if df[c].std() > 0.01
    ]

    if not geo_cols:
        raise ValueError(
            "No numeric GeoTessera columns were found."
        )

    print(
        f"Candidate GeoTessera dimensions: {len(geo_cols):,}"
    )

    # Feature selection, following the original approach.
    selector = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=1,
    )

    selector.fit(
        df[geo_cols],
        df["Amax"],
    )

    importance = pd.Series(
        selector.feature_importances_,
        index=geo_cols,
    )

    geo_cols = (
        importance
        .sort_values(ascending=False)
        .head(N_GEO_FEATURES)
        .index
        .tolist()
    )

    print("Selected GeoTessera dimensions:")
    print(geo_cols)

    del selector, importance
    gc.collect()

    soil_cols = list(
        SOIL_RASTER_PATHS.keys()
    )

    # IMPORTANT: no type_* columns.
    predictor_cols = (
        geo_cols + soil_cols
    )

    df = df.dropna(
        subset=predictor_cols + ["Amax"]
    ).reset_index(drop=True)

    X = df[predictor_cols]
    y = df["Amax"]

    print(
        f"Final training rows: {len(df):,}"
    )

    if (
        "Coords_x" not in df.columns
        or "Coords_y" not in df.columns
    ):
        raise ValueError(
            "Coords_x and Coords_y are required for spatial CV."
        )

    df["lat_bin"] = df["Coords_y"] // 1
    df["lon_bin"] = df["Coords_x"] // 1

    df["spatial_block"] = (
        df["lat_bin"].astype(str)
        + "_"
        + df["lon_bin"].astype(str)
    )

    groups = df["spatial_block"]

    model = RandomForestRegressor(
        n_estimators=1000,
        max_depth=15,
        min_samples_leaf=15,
        min_samples_split=30,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=1,
    )

    print("Running spatial cross-validation...")

    scores = cross_validate(
        model,
        X,
        y,
        cv=GroupKFold(n_splits=5),
        groups=groups,
        scoring={
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        },
        n_jobs=1,
        return_train_score=True,
    )

    print("\n===== Spatial CV =====")
    print(
        "Mean train R2:",
        round(scores["train_r2"].mean(), 4),
    )
    print(
        "Mean test R2:",
        round(scores["test_r2"].mean(), 4),
    )
    print(
        "Mean test RMSE:",
        round(-scores["test_rmse"].mean(), 4),
    )
    print(
        "Mean test MAE:",
        round(-scores["test_mae"].mean(), 4),
    )

    del scores
    gc.collect()

    print("\nFitting final model...")
    model.fit(X, y)

    geo_indices = [
        int(c) for c in geo_cols
    ]

    del X, y, df
    gc.collect()

    return (
        model,
        predictor_cols,
        geo_indices,
    )


# =============================================================================
# OUTPUT GRID
# =============================================================================

def create_output_grid():
    """
    Create a fixed 100 m EPSG:6933 grid covering the forest raster extent.
    """

    print("\n" + "=" * 72)
    print("STEP 2 - CREATE OUTPUT GRID")
    print("=" * 72)

    with rasterio.open(
        FOREST_RASTER
    ) as forest_ds:

        if forest_ds.crs is None:
            raise ValueError(
                "Forest raster has no CRS."
            )

        left, bottom, right, top = transform_bounds(
            forest_ds.crs,
            TARGET_CRS,
            *forest_ds.bounds,
            densify_pts=21,
        )

    resolution = OUTPUT_RESOLUTION

    width = int(
        np.ceil(
            (right - left) / resolution
        )
    )

    height = int(
        np.ceil(
            (top - bottom) / resolution
        )
    )

    transform = from_origin(
        left,
        top,
        resolution,
        resolution,
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    print(f"Output CRS: {TARGET_CRS}")
    print(f"Output resolution: {resolution} m")
    print(f"Output dimensions: {width} x {height}")

    return (
        profile,
        transform,
        width,
        height,
    )


def get_tessera_bbox():
    with rasterio.open(
        FOREST_RASTER
    ) as forest_ds:

        bbox = transform_bounds(
            forest_ds.crs,
            "EPSG:4326",
            *forest_ds.bounds,
            densify_pts=21,
        )

    print("\nTESSERA WGS84 bbox:")
    print(
        f"  west={bbox[0]}, south={bbox[1]}, "
        f"east={bbox[2]}, north={bbox[3]}"
    )

    return bbox


# =============================================================================
# SOIL SAMPLING
# =============================================================================

def sample_soil_rasters(
    soil_datasets,
    tile_crs,
    tile_transform,
    height,
    width,
):
    columns = []

    for _, src_ds in soil_datasets.items():

        destination = np.full(
            (height, width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(
                src_ds,
                1,
            ),
            destination=destination,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=tile_transform,
            dst_crs=tile_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )

        columns.append(
            destination.ravel()
        )

    return np.column_stack(
        columns
    ).astype(
        np.float32,
        copy=False,
    )


# =============================================================================
# OUTPUT WINDOW
# =============================================================================

def output_window_from_bounds(
    bounds,
    output_transform,
    output_width,
    output_height,
):
    """
    Find the integer output-grid window intersecting a prediction strip.
    """

    left, bottom, right, top = bounds

    w = from_bounds(
        left,
        bottom,
        right,
        top,
        transform=output_transform,
    )

    col0 = max(
        0,
        int(np.floor(w.col_off)),
    )
    row0 = max(
        0,
        int(np.floor(w.row_off)),
    )

    col1 = min(
        output_width,
        int(np.ceil(
            w.col_off + w.width
        )),
    )
    row1 = min(
        output_height,
        int(np.ceil(
            w.row_off + w.height
        )),
    )

    if (
        col1 <= col0
        or row1 <= row0
    ):
        return None

    return Window(
        col0,
        row0,
        col1 - col0,
        row1 - row0,
    )


# =============================================================================
# LAZY TESSERA PREDICTION
# =============================================================================

def predict_lazy(
    model,
    predictor_cols,
    geo_indices,
    output_profile,
    output_transform,
    output_width,
    output_height,
):
    """
    Stream TESSERA spatial strips and write them directly to the final
    EPSG:6933 GeoTIFF.
    """

    print("\n" + "=" * 72)
    print("STEP 3 - LAZY TESSERA PREDICTION")
    print("=" * 72)

    bbox = get_tessera_bbox()

    print("Opening GeoTessera...")
    gt = GeoTesseraZarr()

    soil_datasets = {
        name: rasterio.open(path)
        for name, path in SOIL_RASTER_PATHS.items()
    }

    try:

        with rasterio.open(
            OUTPUT_RASTER,
            "w",
            **output_profile,
        ) as output_ds:

            # Initialize the output to NoData in blocks.
            init_h = min(
                512,
                output_height,
            )
            init_w = min(
                512,
                output_width,
            )

            blank = np.full(
                (init_h, init_w),
                np.nan,
                dtype=np.float32,
            )

            for row_off in range(
                0,
                output_height,
                init_h,
            ):
                rows = min(
                    init_h,
                    output_height - row_off,
                )

                for col_off in range(
                    0,
                    output_width,
                    init_w,
                ):
                    cols = min(
                        init_w,
                        output_width - col_off,
                    )

                    output_ds.write(
                        blank[:rows, :cols],
                        1,
                        window=Window(
                            col_off,
                            row_off,
                            cols,
                            rows,
                        ),
                    )

            del blank

            # Current GeoTessera API:
            # iter_region returns block, transform, crs.
            iterator = gt.iter_region(
                bbox,
                year=YEAR,
                strip_rows=STRIP_ROWS,
            )

            for strip_number, (
                tile_data,
                transform,
                crs,
            ) in enumerate(
                iterator,
                start=1,
            ):

                if (
                    tile_data is None
                    or tile_data.size == 0
                ):
                    continue

                if np.isnan(
                    tile_data
                ).all():
                    del tile_data
                    continue

                height, width, channels = (
                    tile_data.shape
                )

                print(
                    f"\nStrip {strip_number}: "
                    f"{height} x {width} x {channels}"
                )

                # -------------------------------------------------------------
                # Extract only selected GeoTessera dimensions.
                # -------------------------------------------------------------

                geo_pixels = (
                    tile_data
                    .reshape(-1, channels)
                    [:, geo_indices]
                    .copy()
                )

                del tile_data
                gc.collect()

                # -------------------------------------------------------------
                # Add continuous environmental predictors.
                # -------------------------------------------------------------

                soil_pixels = sample_soil_rasters(
                    soil_datasets,
                    crs,
                    transform,
                    height,
                    width,
                )

                # No type_* predictors.
                X_strip = np.hstack(
                    [
                        geo_pixels,
                        soil_pixels,
                    ]
                ).astype(
                    np.float32,
                    copy=False,
                )

                del geo_pixels
                del soil_pixels
                gc.collect()

                if (
                    X_strip.shape[1]
                    != len(predictor_cols)
                ):
                    raise ValueError(
                        "Prediction feature count does not match training. "
                        f"Prediction={X_strip.shape[1]}, "
                        f"Training={len(predictor_cols)}"
                    )

                # -------------------------------------------------------------
                # Predict only finite rows.
                # -------------------------------------------------------------

                valid = np.isfinite(
                    X_strip
                ).all(
                    axis=1
                )

                prediction = np.full(
                    X_strip.shape[0],
                    np.nan,
                    dtype=np.float32,
                )

                if valid.any():
                    prediction[valid] = (
                        model.predict(
                            X_strip[valid]
                        ).astype(
                            np.float32,
                            copy=False,
                        )
                    )

                del X_strip
                del valid
                gc.collect()

                prediction = prediction.reshape(
                    height,
                    width,
                )

                # -------------------------------------------------------------
                # Aggregate by factor 10.
                # -------------------------------------------------------------

                h_trim = (
                    height
                    // AGGREGATION_FACTOR
                ) * AGGREGATION_FACTOR

                w_trim = (
                    width
                    // AGGREGATION_FACTOR
                ) * AGGREGATION_FACTOR

                if (
                    h_trim == 0
                    or w_trim == 0
                ):
                    del prediction
                    continue

                prediction = (
                    prediction[
                        :h_trim,
                        :w_trim
                    ]
                    .reshape(
                        h_trim // AGGREGATION_FACTOR,
                        AGGREGATION_FACTOR,
                        w_trim // AGGREGATION_FACTOR,
                        AGGREGATION_FACTOR,
                    )
                    .mean(
                        axis=(1, 3)
                    )
                    .astype(
                        np.float32
                    )
                )

                prediction_transform = rasterio.Affine(
                    transform.a * AGGREGATION_FACTOR,
                    transform.b,
                    transform.c,
                    transform.d,
                    transform.e * AGGREGATION_FACTOR,
                    transform.f,
                )

                pred_h, pred_w = (
                    prediction.shape
                )

                bounds = array_bounds(
                    pred_h,
                    pred_w,
                    prediction_transform,
                )

                window = output_window_from_bounds(
                    bounds,
                    output_transform,
                    output_width,
                    output_height,
                )

                if window is None:
                    del prediction
                    continue

                destination = np.full(
                    (
                        int(window.height),
                        int(window.width),
                    ),
                    np.nan,
                    dtype=np.float32,
                )

                destination_transform = (
                    rasterio.windows.transform(
                        window,
                        output_transform,
                    )
                )

                # -------------------------------------------------------------
                # Reproject this one strip into its final output window.
                # -------------------------------------------------------------

                reproject(
                    source=prediction,
                    destination=destination,
                    src_transform=prediction_transform,
                    src_crs=crs,
                    dst_transform=destination_transform,
                    dst_crs=TARGET_CRS,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )

                del prediction
                gc.collect()

                output_ds.write(
                    destination,
                    1,
                    window=window,
                )

                del destination
                gc.collect()

                print(
                    f"  wrote row={window.row_off}, "
                    f"col={window.col_off}, "
                    f"h={window.height}, "
                    f"w={window.width}"
                )

    finally:

        for ds in soil_datasets.values():
            ds.close()

    print(
        "\nLazy TESSERA prediction complete."
    )


# =============================================================================
# FOREST MASK
# =============================================================================

def apply_forest_mask():
    """
    Apply:
        forest == 0 -> retain prediction
        forest == 1 -> NoData
    """

    print("\n" + "=" * 72)
    print("STEP 4 - APPLY FOREST MASK")
    print("=" * 72)

    with rasterio.open(
        OUTPUT_RASTER,
        "r+",
    ) as prediction_ds:

        with rasterio.open(
            FOREST_RASTER
        ) as forest_ds:

            for _, window in prediction_ds.block_windows(
                1
            ):

                prediction = prediction_ds.read(
                    1,
                    window=window,
                )

                forest = np.ones(
                    prediction.shape,
                    dtype=np.uint8,
                )

                reproject(
                    source=rasterio.band(
                        forest_ds,
                        1,
                    ),
                    destination=forest,
                    src_transform=forest_ds.transform,
                    src_crs=forest_ds.crs,
                    dst_transform=prediction_ds.window_transform(
                        window
                    ),
                    dst_crs=prediction_ds.crs,
                    resampling=Resampling.nearest,
                )

                # 0 = forest, therefore only keep 0.
                prediction[
                    forest != FOREST_VALUE
                ] = np.nan

                prediction_ds.write(
                    prediction.astype(
                        np.float32
                    ),
                    1,
                    window=window,
                )

                del prediction
                del forest

    print("Forest mask applied.")
    print("0 = forest: prediction retained.")
    print("1 = non-forest: prediction set to NoData.")


# =============================================================================
# MAIN
# =============================================================================

def main():

    validate_paths()

    (
        model,
        predictor_cols,
        geo_indices,
    ) = train_model()

    (
        output_profile,
        output_transform,
        output_width,
        output_height,
    ) = create_output_grid()

    predict_lazy(
        model=model,
        predictor_cols=predictor_cols,
        geo_indices=geo_indices,
        output_profile=output_profile,
        output_transform=output_transform,
        output_width=output_width,
        output_height=output_height,
    )

    apply_forest_mask()

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(
        f"Final GeoTIFF:\n{OUTPUT_RASTER}"
    )


if __name__ == "__main__":
    main()
