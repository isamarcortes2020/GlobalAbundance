# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:15:31 2026

@author: cenv1124
"""

# -*- coding: utf-8 -*-
"""
TESSERA (AWS Icechunk, v1.1) -> Random Forest -> forest-only STRIP GeoTIFFs.

Changes from the GeoTessera version
-----------------------------------
- Embeddings are read from the public dClimate Icechunk store on S3
  (s3://tessera-embeddings/v1.1/dclimate.icechunk), one UTM-zone group at a time.
- Only the selected GeoTessera bands are read and dequantised
  (embeddings int8 * per-pixel float32 `scales`).
- Forest mask: 1 = forest (kept), 0 = non-forest (NoData).
- Strip bounds are converted to the output CRS before the output window is found
  (this was why nothing was saved before).
- Strips with nothing to save are reported and NOT written to the checkpoint
  unless they are genuinely empty (no forest / no data).
- Strips with no forest are skipped BEFORE downloading any embeddings.
- STRIP_ROWS is a multiple of AGGREGATION_FACTOR so strips tile without gaps.
- Adjacent UTM zone groups overlap, so each pixel is only kept inside its
  nominal zone (lon range of the zone + hemisphere).
"""

import os
# Parallelism is done with worker processes, so force libraries to 1 thread each
# (schedulers such as SLURM often set OMP_NUM_THREADS to the CPU count, which
# would oversubscribe the node).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import gc
import json
import multiprocessing
import re
import socket
import sys
import threading
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import xarray as xr
import icechunk
import joblib
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_origin, array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window, from_bounds

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAINING_CSV = (
    "R:/GlobalDataset/TraitsCombinedWithGeoTessera/FinalizedTesseraData/AmaxCombined.csv"
)

FOREST_RASTER = os.environ.get("FOREST_RASTER") or "C:/Users/cenv1124/Downloads/foresttest.tif"

STRIP_OUTPUT_DIR = "C:/Users/cenv1124/Downloads/Amax_2024_forest_strips"

FINAL_MOSAIC = (
    "C:/Users/cenv1124/Downloads/Amax_2024_forest_only_EPSG6933.tif"
)

# --- Several polygons in ONE output folder -----------------------------------
# Give each polygon its own RUN_TAG (letters, digits, hyphens only; no
# underscores), e.g. "gabon". Strips are then named strip_<tag>_<key>.tif and
# each polygon keeps its own checkpoint / lock / run_meta files, so polygons
# never skip, overwrite or block each other, and you can merge the whole folder.
# A tagged run also snaps the output grid to a global 100 m lattice (anchored
# at 0,0 in EPSG:6933) so strips from different polygons line up exactly.
# Leave "" for the legacy single-polygon behaviour. Env var RUN_TAG overrides.
RUN_TAG = os.environ.get("RUN_TAG", "")
if not re.fullmatch(r"[A-Za-z0-9-]*", RUN_TAG):
    raise SystemExit("RUN_TAG may only contain letters, digits and hyphens.")
SNAP_OUTPUT_GRID = bool(RUN_TAG)

TARGET_CRS = "EPSG:6933"
YEAR = 2024

# Forest raster: 1 = forest (keep), anything else = non-forest.
FOREST_VALUE = 1

# Optional small test window in WGS84 (west, south, east, north), or None for
# the whole forest raster. It is intersected with the raster extent.
#   (25.0, 1.0, 25.2, 1.2)    -> one zone (35N), good first test
#   (23.9, -0.1, 24.1, 0.1)   -> crosses the 34/35 zone seam AND the equator
# Env var TEST_BBOX="west,south,east,north" overrides, e.g. TEST_BBOX=50.05,-15.21,50.25,-15.01
_tb = os.environ.get("TEST_BBOX")
TEST_BBOX = tuple(float(v) for v in _tb.split(",")) if _tb else None
if TEST_BBOX is not None and len(TEST_BBOX) != 4:
    raise SystemExit("TEST_BBOX needs 4 numbers: west,south,east,north")

# AWS Icechunk store
ICECHUNK_BUCKET = "tessera-embeddings"
ICECHUNK_PREFIX = "v1.1/dclimate.icechunk"
ICECHUNK_REGION = "us-west-2"

TESSERA_RESOLUTION = 10.0
AGGREGATION_FACTOR = 10
OUTPUT_RESOLUTION = TESSERA_RESOLUTION * AGGREGATION_FACTOR

# Rows of 10 m pixels per strip. MUST be a multiple of AGGREGATION_FACTOR,
# otherwise each strip loses rows at the bottom and leaves gaps.
STRIP_ROWS = 120
assert STRIP_ROWS % AGGREGATION_FACTOR == 0

# Fraction of valid 10 m pixels required in a 10x10 block to keep the mean.
MIN_VALID_FRACTION = 0.5

# Optional speed-up: predict only every PIXEL_STRIDE-th pixel inside each
# 10x10 block (rows and columns) and average those. 1 = every pixel (exact).
#   2 -> 25 px/block (~4x faster), 3 -> 16 px/block (~6x), 4 -> 9 px/block (~11x)
PIXEL_STRIDE = 1

# Parallelism: number of worker processes. Each needs roughly 1-2 GB of RAM
# with COL_BLOCK = 10000. Start with 4 and watch memory.
def _default_workers():
    """CPUs actually allocated to this job (SLURM / cgroup aware)."""
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n and n.isdigit():
        return int(n)
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


# Env var WORKERS overrides. ~2 GB RAM each; must not exceed the CPUs you were given.
WORKERS = int(os.environ["WORKERS"]) if os.environ.get("WORKERS") else 24

# Columns (10 m pixels) processed at a time inside a strip. Keeps memory
# small. Must be a multiple of AGGREGATION_FACTOR.
COL_BLOCK = 10000

# Write the checkpoint after this many finished strips (or every 30 s).
CHECKPOINT_EVERY = 25
assert COL_BLOCK % AGGREGATION_FACTOR == 0

# Sharding: run several jobs at once, each doing a disjoint share of the strips.
# Set NUM_SHARDS and SHARD_INDEX (0-based) as environment variables, or just
# submit a SLURM array job (--array=0-7) and they are picked up automatically.
def _shard_from_env():
    n, i = os.environ.get("NUM_SHARDS"), os.environ.get("SHARD_INDEX")
    if n and i:
        return int(i), int(n)
    cnt = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    if cnt and tid:
        lo = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
        return int(tid) - lo, int(cnt)
    return 0, 1


SHARD_INDEX, NUM_SHARDS = _shard_from_env()
if not (0 <= SHARD_INDEX < NUM_SHARDS):
    raise SystemExit(f"Bad shard: index {SHARD_INDEX} of {NUM_SHARDS}")


def shard_of(key):
    """Stable (process-independent) shard assignment for a strip key."""
    return zlib.crc32(key.encode("utf-8")) % NUM_SHARDS


# The trained model is saved here and reused on later runs (skips training/CV).
MODEL_CACHE = "C:/Users/cenv1124/Downloads/Amax_rf_model.joblib"
USE_MODEL_CACHE = True
RUN_CV = True    # set False to skip the (slow) spatial cross-validation

# Set True ONLY when resuming a run started with the single-process script,
# using the same TEST_BBOX / STRIP_ROWS / settings in the same strip folder.
RESUME_FROM_UNVERSIONED_RUN = False

# The Python mosaic is slow for thousands of strips. Merge in QGIS instead
# (Build Virtual Raster) or set True to use build_final_mosaic().
BUILD_MOSAIC = False

N_GEO_FEATURES = 20

MAX_TESSERA_RETRIES = 5
RETRY_WAIT_SECONDS = 10

SOIL_RASTER_PATHS = {
    "CEC": "R:/GlobalDataset/EnvironmentalData/cec_avg.tif",
    "Clay": "R:/GlobalDataset/EnvironmentalData/clay_avg.tif",
    "Sand": "R:/GlobalDataset/EnvironmentalData/sand_avg.tif",
    "pH": "R:/GlobalDataset/EnvironmentalData/pH_avg.tif",
    "slope": "R:/GlobalDataset/EnvironmentalData/SRTM_global.tif",
    "mcwd_mean": "R:/GlobalDataset/EnvironmentalData/MCWD_mean_1988_2017.tif",
    "tmax_mean": "R:/GlobalDataset/EnvironmentalData/Tmax_mean_1988_2017.tif",
}


# =============================================================================
# VALIDATION
# =============================================================================

def validate_paths():
    paths = {
        "Training CSV": TRAINING_CSV,
        "Forest raster": FOREST_RASTER,
    }
    paths.update({
        f"Soil raster: {name}": path
        for name, path in SOIL_RASTER_PATHS.items()
    })

    for label, path in paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} does not exist:\n{path}")
        print(f"OK: {label}")

    Path(STRIP_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# =============================================================================
# TRAINING  (unchanged)
# =============================================================================

def train_model():
    print("\n" + "=" * 72)
    print("STEP 1 - TRAIN MODEL")
    print("=" * 72)

    df = pd.read_csv(TRAINING_CSV)
    print(f"Original rows: {len(df):,}")

    df = df[df["label"].isin([1, 3, 5])].copy()
    print(f"Rows after label filter [1, 3, 5]: {len(df):,}")

    df = df.dropna(subset=["0", "Amax"]).copy()

    geo_cols = [c for c in df.columns if str(c).isdigit()]
    geo_cols = [c for c in geo_cols if df[c].std() > 0.01]

    if not geo_cols:
        raise ValueError("No numeric GeoTessera columns were found.")

    print(f"Candidate GeoTessera dimensions: {len(geo_cols):,}")

    selector = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=1,
    )
    selector.fit(df[geo_cols], df["Amax"])

    importance = pd.Series(selector.feature_importances_, index=geo_cols)
    geo_cols = (
        importance.sort_values(ascending=False)
        .head(N_GEO_FEATURES).index.tolist()
    )

    print("Selected GeoTessera dimensions:")
    print(geo_cols)

    del selector, importance
    gc.collect()

    soil_cols = list(SOIL_RASTER_PATHS.keys())
    predictor_cols = geo_cols + soil_cols

    df = df.dropna(subset=predictor_cols + ["Amax"]).reset_index(drop=True)

    X = df[predictor_cols]
    y = df["Amax"]

    print(f"Final training rows: {len(df):,}")

    if "Coords_x" not in df.columns or "Coords_y" not in df.columns:
        raise ValueError("Coords_x and Coords_y are required for spatial CV.")

    df["lat_bin"] = df["Coords_y"] // 1
    df["lon_bin"] = df["Coords_x"] // 1
    df["spatial_block"] = (
        df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
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

    if RUN_CV:
        print("Running spatial cross-validation...")

        scores = cross_validate(
            model, X, y,
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
        print("Mean train R2:", round(scores["train_r2"].mean(), 4))
        print("Mean test R2:", round(scores["test_r2"].mean(), 4))
        print("Mean test RMSE:", round(-scores["test_rmse"].mean(), 4))
        print("Mean test MAE:", round(-scores["test_mae"].mean(), 4))

        del scores
        gc.collect()

    print("\nFitting final model...")
    model.fit(X, y)

    geo_indices = [int(c) for c in geo_cols]

    del X, y, df
    gc.collect()

    return model, predictor_cols, geo_indices


# =============================================================================
# OUTPUT GRID
# =============================================================================

def create_output_grid():
    print("\n" + "=" * 72)
    print("STEP 2 - DEFINE OUTPUT GRID")
    print("=" * 72)

    with rasterio.open(FOREST_RASTER) as forest_ds:
        if forest_ds.crs is None:
            raise ValueError("Forest raster has no CRS.")

    left, bottom, right, top = transform_bounds(
        "EPSG:4326", TARGET_CRS, *get_wgs84_bbox(verbose=False), densify_pts=21,
    )

    resolution = OUTPUT_RESOLUTION

    if SNAP_OUTPUT_GRID:
        # Global lattice: every polygon's grid is a sub-window of the same grid.
        left = np.floor(left / resolution) * resolution
        bottom = np.floor(bottom / resolution) * resolution
        right = np.ceil(right / resolution) * resolution
        top = np.ceil(top / resolution) * resolution

    width = int(np.ceil((right - left) / resolution))
    height = int(np.ceil((top - bottom) / resolution))
    transform = from_origin(left, top, resolution, resolution)

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

    return profile, transform, width, height


def get_wgs84_bbox(verbose=True):
    with rasterio.open(FOREST_RASTER) as forest_ds:
        west, south, east, north = transform_bounds(
            forest_ds.crs, "EPSG:4326", *forest_ds.bounds, densify_pts=21,
        )

    if TEST_BBOX is not None:
        tw, ts, te, tn = TEST_BBOX
        west, south = max(west, tw), max(south, ts)
        east, north = min(east, te), min(north, tn)
        if west >= east or south >= north:
            raise ValueError("TEST_BBOX does not overlap the forest raster.")

    if verbose:
        print("\nWGS84 bbox used:")
        print(f"  west={west}, south={south}, east={east}, north={north}")

    return (west, south, east, north)


# =============================================================================
# TESSERA ICECHUNK STORE
# =============================================================================

def is_missing_group(exc):
    return (
        isinstance(exc, (KeyError, FileNotFoundError))
        or "NotFound" in type(exc).__name__
    )


def ensure_pcodec():
    """The 'scales' array is PCodec-compressed. Fail fast, and say which Python."""
    import numcodecs
    try:
        numcodecs.get_codec({"id": "pcodec"})
        return
    except Exception:
        pass
    try:
        from numcodecs.pcodec import PCodec
        numcodecs.register_codec(PCodec)
        numcodecs.get_codec({"id": "pcodec"})
    except Exception as exc:
        raise RuntimeError(
            "The 'pcodec' codec is not available in this Python:\n"
            f"  {sys.executable}\n"
            f"  ({type(exc).__name__}: {exc})\n"
            "Install it into THIS interpreter with:\n"
            f'  "{sys.executable}" -m pip install -U "numcodecs[pcodec]"'
        ) from exc


def is_fatal_error(exc):
    """Errors that retrying can never fix."""
    return "UnknownCodec" in type(exc).__name__


class TesseraStore:
    """Read-only handle on the public dClimate Icechunk store, with retries."""

    def __init__(self):
        self._open()

    def _open(self):
        ensure_pcodec()
        storage = icechunk.s3_storage(
            bucket=ICECHUNK_BUCKET,
            prefix=ICECHUNK_PREFIX,
            region=ICECHUNK_REGION,
            anonymous=True,
        )
        repo = icechunk.Repository.open(storage)
        self.session = repo.readonly_session(branch="main")
        self._groups = {}

    def reopen(self):
        self._groups = {}
        self.session = None
        gc.collect()
        self._open()

    def group(self, name):
        if name not in self._groups:
            self._groups[name] = xr.open_zarr(
                self.session.store,
                group=name,
                consolidated=False,
                decode_coords="all",
                chunks=None,
            )
        return self._groups[name]

    def call(self, fn, what):
        """Run fn(); on a remote failure wait, reopen the store and retry."""
        for attempt in range(MAX_TESSERA_RETRIES + 1):
            try:
                return fn()
            except Exception as exc:
                if is_missing_group(exc) or is_fatal_error(exc):
                    raise
                print("\n" + "!" * 72)
                print(f"TESSERA error while {what}:")
                print(f"{type(exc).__name__}: {exc}")
                print("!" * 72)

                if attempt >= MAX_TESSERA_RETRIES:
                    print("Maximum retries reached. Completed strips are kept.")
                    raise

                print(
                    f"Retry {attempt + 1}/{MAX_TESSERA_RETRIES} "
                    f"in {RETRY_WAIT_SECONDS} s..."
                )
                time.sleep(RETRY_WAIT_SECONDS)
                try:
                    self.reopen()
                except Exception as reopen_exc:
                    print(f"Reopen failed: {reopen_exc}")


def check_dataset(ds, group):
    needed_dims = {"time", "easting", "northing", "band"}
    ok = (
        "embeddings" in ds.variables
        and "scales" in ds.variables
        and needed_dims <= set(ds["embeddings"].dims)
        and {"time", "easting", "northing"} <= set(ds["scales"].dims)
    )
    if not ok:
        raise ValueError(
            f"Group {group} does not look like the expected layout "
            f"(embeddings/scales with time, easting, northing, band).\n{ds}"
        )


def year_index(ds, year):
    tvals = ds["time"].values
    if np.issubdtype(tvals.dtype, np.datetime64):
        years = pd.DatetimeIndex(tvals).year.values
    else:
        years = np.asarray(tvals).astype(int)
    hits = np.where(years == year)[0]
    return int(hits[0]) if len(hits) else None


def warn_if_year_incomplete(ds, group, year):
    yc = ds.attrs.get("years_complete")
    if yc is None:
        print(f"  (no 'years_complete' attribute on {group}; can't verify year)")
        return
    try:
        if year not in [int(y) for y in yc]:
            print(
                f"  WARNING: {year} is not in years_complete for {group}; "
                f"it may read back as fill (NaN)."
            )
    except Exception:
        print(f"  years_complete on {group}: {yc}")


def read_strip(ds, t_idx, geo_indices, r0, r1, c0, c1, north_ascending):
    """
    Read one strip, only the selected bands, dequantised to float32.
    Returns (rows, cols, n_bands) with row 0 = northernmost row.
    """
    sl = dict(northing=slice(r0, r1), easting=slice(c0, c1))

    emb = (
        ds["embeddings"]
        .isel(time=t_idx, band=list(geo_indices), **sl)
        .transpose("northing", "easting", "band")
        .values
    )
    scales = (
        ds["scales"]
        .isel(time=t_idx, **sl)
        .transpose("northing", "easting")
        .values
    )

    data = emb.astype(np.float32) * scales.astype(np.float32)[:, :, None]

    if north_ascending:
        data = data[::-1]

    return np.ascontiguousarray(data)


# =============================================================================
# ZONE HELPERS
# =============================================================================

def zone_groups_for_bbox(bbox):
    west, south, east, north = bbox
    z0 = min(max(int(np.floor((west + 180) / 6)) + 1, 1), 60)
    z1 = min(max(int(np.floor((east + 180) / 6)) + 1, 1), 60)

    hemis = []
    if north >= 0:
        hemis.append("N")
    if south < 0:
        hemis.append("S")

    return [(z, h) for z in range(z0, z1 + 1) for h in hemis]


def zone_crs(zone, hemi):
    return f"EPSG:{32600 + zone if hemi == 'N' else 32700 + zone}"


def nominal_zone_mask(transform, crs, h, w, zone, hemi):
    """True where the pixel centre lies in this zone's nominal lon range/hemisphere."""
    xs = transform.c + (np.arange(w) + 0.5) * transform.a
    ys = transform.f + (np.arange(h) + 0.5) * transform.e
    X, Y = np.meshgrid(xs, ys)

    lons, lats = warp_transform(
        crs, "EPSG:4326", X.ravel().tolist(), Y.ravel().tolist(),
    )
    lons = np.asarray(lons).reshape(h, w)
    lats = np.asarray(lats).reshape(h, w)

    west = (zone - 1) * 6 - 180
    east = west + 6

    mask = (lons >= west) & (lons < east)
    mask &= (lats >= 0) if hemi == "N" else (lats < 0)
    return mask


# =============================================================================
# SOIL SAMPLING
# =============================================================================

def sample_soil_rasters(soil_datasets, tile_crs, tile_transform, height, width):
    columns = []

    for _, src_ds in soil_datasets.items():
        destination = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src_ds, 1),
            destination=destination,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=tile_transform,
            dst_crs=tile_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        columns.append(destination.ravel())

    return np.column_stack(columns).astype(np.float32, copy=False)


# =============================================================================
# FOREST HELPERS   (1 = forest, 0 = non-forest)
# =============================================================================

def strip_has_forest(forest_ds, agg_transform, crs, agg_h, agg_w):
    """
    Cheap pre-check on the 100 m strip grid. Uses 'max' resampling so a single
    forest pixel inside a block counts (never wrongly skips a strip).
    """
    arr = np.zeros((agg_h, agg_w), dtype=np.uint8)

    reproject(
        source=rasterio.band(forest_ds, 1),
        destination=arr,
        src_transform=forest_ds.transform,
        src_crs=forest_ds.crs,
        dst_transform=agg_transform,
        dst_crs=crs,
        resampling=Resampling.max,
        dst_nodata=0,
    )
    return bool((arr == FOREST_VALUE).any())


# =============================================================================
# OUTPUT WINDOW
# =============================================================================

def output_window_from_bounds(bounds, output_transform, output_width, output_height):
    left, bottom, right, top = bounds

    w = from_bounds(left, bottom, right, top, transform=output_transform)

    col0 = max(0, int(np.floor(w.col_off)))
    row0 = max(0, int(np.floor(w.row_off)))
    col1 = min(output_width, int(np.ceil(w.col_off + w.width)))
    row1 = min(output_height, int(np.ceil(w.row_off + w.height)))

    if col1 <= col0 or row1 <= row0:
        return None

    return Window(col0, row0, col1 - col0, row1 - row0)


# =============================================================================
# STRIP FILES / CHECKPOINT   (keys look like "33N_0007")
# =============================================================================

def _tagged(base):
    return f"{base}__{RUN_TAG}" if RUN_TAG else base


def _shard_suffix():
    return "" if NUM_SHARDS == 1 else f".shard{SHARD_INDEX}of{NUM_SHARDS}"


def _family(base, ext):
    """Files belonging to THIS polygon (tag) for a given base name."""
    rx = re.compile(
        rf"^{re.escape(_tagged(base))}(\.shard\d+of\d+)?\.{ext}$"
    )
    return [
        p for p in sorted(Path(STRIP_OUTPUT_DIR).iterdir()) if rx.match(p.name)
    ]


def strip_prefix():
    return f"strip_{RUN_TAG}_" if RUN_TAG else "strip_"


_LEGACY_KEY = re.compile(r"^\d{2}[NS]_\d+$")


def strip_files_for_this_run():
    """Strip TIFFs of THIS polygon only (untagged = legacy names like 35N_0001)."""
    files = sorted(Path(STRIP_OUTPUT_DIR).glob(f"{strip_prefix()}*.tif"))
    if RUN_TAG:
        return files
    return [f for f in files if _LEGACY_KEY.match(f.stem[len("strip_"):])]


def strip_path(key):
    return Path(STRIP_OUTPUT_DIR) / f"{strip_prefix()}{key}.tif"


def checkpoint_path():
    """This shard's own checkpoint file (per polygon tag)."""
    return Path(STRIP_OUTPUT_DIR) / (
        f"{_tagged('completed_strips')}{_shard_suffix()}.json"
    )


def all_checkpoint_paths():
    return _family("completed_strips", "json")


def load_json_keys(path):
    if not Path(path).exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(str(v) for v in json.load(f))
    except Exception:
        print(f"WARNING: could not read checkpoint {path}.")
        return set()


def load_completed_strips():
    """Union of EVERY shard's checkpoint (so any sharding scheme can resume)."""
    done = set()
    for path in all_checkpoint_paths():
        done |= load_json_keys(path)
    return done


def save_completed_strips(keys):
    """
    Best-effort checkpoint of THIS shard's keys. Unique temp file per process,
    retries on filesystem hiccups, never crashes the run (strip TIFFs also
    record progress).
    """
    path = checkpoint_path()
    temp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")

    for attempt in range(4):
        try:
            with open(temp, "w", encoding="utf-8") as f:
                json.dump(sorted(keys), f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, path)
            return True
        except OSError as exc:
            print(
                f"WARNING: checkpoint write failed ({exc}); "
                f"retry {attempt + 1}/4",
                flush=True,
            )
            time.sleep(1 + attempt)

    print("WARNING: could not write the checkpoint; continuing.", flush=True)
    return False


def existing_completed_strips():
    completed = load_completed_strips()
    for path in strip_files_for_this_run():
        completed.add(path.stem[len(strip_prefix()):])
    return completed


# --- guard against two runs using the same folder ---------------------------

LOCK_STALE_SECONDS = 1800   # a lock not refreshed for this long is ignored


def lock_path():
    return Path(STRIP_OUTPUT_DIR) / f"{_tagged('run')}{_shard_suffix()}.lock"


def acquire_run_lock():
    lp = lock_path()
    info = (
        f"host={socket.gethostname()} pid={os.getpid()} "
        f"slurm_job={os.environ.get('SLURM_JOB_ID', '-')}"
    )

    # An UNSHARDED run must not start while any shard is active.
    if NUM_SHARDS == 1:
        for other in _family("run", "lock"):
            if other != lp and time.time() - other.stat().st_mtime < LOCK_STALE_SECONDS:
                raise RuntimeError(
                    f"A sharded run is active ({other.name}); an unsharded run "
                    "would duplicate its work. Wait for it, or delete the lock "
                    "if nothing is running."
                )

    try:
        fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(info)
    except FileExistsError:
        age = time.time() - lp.stat().st_mtime
        if age < LOCK_STALE_SECONDS:
            try:
                other = lp.read_text()
            except Exception:
                other = "?"
            raise RuntimeError(
                "Another run seems to be active in this strip folder "
                f"(lock refreshed {age:.0f} s ago by: {other}).\n"
                "Two runs in one folder corrupt each other's checkpoint.\n"
                "Check `squeue -u $USER`. If nothing is running, delete:\n"
                f"  {lp}"
            )
        print(f"Found a stale lock ({age / 60:.0f} min old); taking over.")
        lp.write_text(info)

    stop = threading.Event()

    def beat():
        while not stop.wait(60):
            try:
                os.utime(lp, None)
            except OSError:
                pass

    threading.Thread(target=beat, daemon=True).start()
    return stop


def release_run_lock(stop):
    stop.set()
    try:
        lock_path().unlink()
    except OSError:
        pass


# =============================================================================
# SAVE ONE STRIP
# =============================================================================

def write_strip_tif(
    prediction,
    prediction_transform,
    prediction_crs,
    output_window,
    output_transform,
    forest_ds,
    key,
):
    """
    Reproject a 100 m prediction strip to the output grid, keep forest pixels
    only (forest == 1), and save it. Returns the path, or None if nothing valid.
    """
    out_height = int(output_window.height)
    out_width = int(output_window.width)

    destination = np.full((out_height, out_width), np.nan, dtype=np.float32)

    destination_transform = rasterio.windows.transform(
        output_window, output_transform,
    )

    reproject(
        source=prediction,
        destination=destination,
        src_transform=prediction_transform,
        src_crs=prediction_crs,
        dst_transform=destination_transform,
        dst_crs=TARGET_CRS,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    # Forest mask on this exact output window. Outside = 0 (non-forest).
    forest = np.zeros((out_height, out_width), dtype=np.uint8)

    reproject(
        source=rasterio.band(forest_ds, 1),
        destination=forest,
        src_transform=forest_ds.transform,
        src_crs=forest_ds.crs,
        dst_transform=destination_transform,
        dst_crs=TARGET_CRS,
        resampling=Resampling.nearest,
        dst_nodata=0,
    )

    destination[forest != FOREST_VALUE] = np.nan

    valid_count = int(np.isfinite(destination).sum())

    if valid_count == 0:
        print("  No forest prediction pixels after masking; nothing written.")
        del destination, forest
        return None

    output_path = strip_path(key)

    profile = {
        "driver": "GTiff",
        "height": out_height,
        "width": out_width,
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": destination_transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    # Write to a temp name, then rename: a crash mid-write can never leave a
    # partial strip_*.tif that a later run would mistake for a finished strip.
    temp_path = output_path.with_suffix(".tmp")

    with rasterio.open(temp_path, "w", **profile) as dst:
        dst.write(destination, 1)

    os.replace(temp_path, output_path)

    print(f"  SAVED: {output_path}")
    print(f"  Size: {out_width} x {out_height}")
    print(f"  Forest prediction pixels: {valid_count:,}")

    del destination, forest
    gc.collect()

    return output_path


# =============================================================================
# BLOCK PREDICTION + STRIP FINISHING
# =============================================================================

def predict_block(tile_data, transform, crs, model, predictor_cols, soil_datasets):
    """
    tile_data: (rows, cols, N_GEO_FEATURES) float32, already band-selected.
    Returns the 100 m aggregated prediction (rows//F, cols//F) float32,
    or None if there is nothing to predict.
    """
    height, width, channels = tile_data.shape
    F = AGGREGATION_FACTOR
    S = PIXEL_STRIDE

    if np.isnan(tile_data).all():
        return None

    h_trim = (height // F) * F
    w_trim = (width // F) * F
    if h_trim == 0 or w_trim == 0:
        return None

    geo_pixels = tile_data.reshape(-1, channels)

    # Optional sub-sampling: only every S-th pixel inside each FxF block.
    if S > 1:
        rr = (np.arange(height) % F) % S == 0
        cc = (np.arange(width) % F) % S == 0
        sample_idx = np.flatnonzero((rr[:, None] & cc[None, :]).ravel())
        per_block = len(range(0, F, S)) ** 2
    else:
        sample_idx = None
        per_block = F * F

    soil_pixels = sample_soil_rasters(soil_datasets, crs, transform, height, width)

    if sample_idx is not None:
        X = np.hstack([geo_pixels[sample_idx], soil_pixels[sample_idx]])
    else:
        X = np.hstack([geo_pixels, soil_pixels])
    X = X.astype(np.float32, copy=False)
    del geo_pixels, soil_pixels

    if X.shape[1] != len(predictor_cols):
        raise ValueError(
            "Prediction feature count does not match training. "
            f"Prediction={X.shape[1]}, Training={len(predictor_cols)}"
        )

    valid = np.isfinite(X).all(axis=1)
    prediction = np.full(height * width, np.nan, dtype=np.float32)

    if valid.any():
        preds = model.predict(
            pd.DataFrame(X[valid], columns=predictor_cols)
        ).astype(np.float32, copy=False)

        if sample_idx is not None:
            prediction[sample_idx[valid]] = preds
        else:
            prediction[np.flatnonzero(valid)] = preds
        del preds

    del X, valid
    prediction = prediction.reshape(height, width)

    blocks = prediction[:h_trim, :w_trim].reshape(
        h_trim // F, F, w_trim // F, F,
    )
    del prediction

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        agg = np.nanmean(blocks, axis=(1, 3)).astype(np.float32)

    counts = np.isfinite(blocks).sum(axis=(1, 3))
    agg[counts < MIN_VALID_FRACTION * per_block] = np.nan
    del blocks, counts

    return agg


def finish_strip(
    key, agg, agg_transform, crs, zone, hemi,
    forest_ds, output_transform, output_width, output_height,
):
    """Zone-mask, reproject to the output grid, forest-mask and save. Returns a status."""
    agg_h, agg_w = agg.shape

    mask = nominal_zone_mask(agg_transform, crs, agg_h, agg_w, zone, hemi)
    agg[~mask] = np.nan

    if not np.isfinite(agg).any():
        print(f"Strip {key}: no valid predictions inside nominal zone; nothing written.")
        return "empty"

    bounds = array_bounds(agg_h, agg_w, agg_transform)
    bounds = transform_bounds(crs, TARGET_CRS, *bounds, densify_pts=21)

    window = output_window_from_bounds(
        bounds, output_transform, output_width, output_height,
    )

    if window is None:
        print(f"Strip {key}: outside output grid; nothing written.")
        return "empty"

    path = write_strip_tif(
        prediction=agg,
        prediction_transform=agg_transform,
        prediction_crs=crs,
        output_window=window,
        output_transform=output_transform,
        forest_ds=forest_ds,
        key=key,
    )

    return "saved" if path is not None else "empty"


# =============================================================================
# RUN SIGNATURE  (protects against mixing strips from different runs)
# =============================================================================

def run_signature(bbox):
    sig = {
        "bbox": [round(float(v), 6) for v in bbox],
        "year": YEAR,
        "strip_rows": STRIP_ROWS,
        "pixel_stride": PIXEL_STRIDE,
        "min_valid_fraction": MIN_VALID_FRACTION,
        "forest_raster": os.path.basename(str(FOREST_RASTER).replace("\\", "/")),
        "forest_value": FOREST_VALUE,
    }
    if SNAP_OUTPUT_GRID:
        sig["snap_grid"] = True
    return sig


def check_run_meta(bbox):
    meta_path = Path(STRIP_OUTPUT_DIR) / f"{_tagged('run_meta')}.json"
    sig = run_signature(bbox)

    old = None
    for attempt in range(3):          # another shard may be writing it right now
        if not meta_path.exists():
            break
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            break
        except Exception:
            time.sleep(1)

    if old is not None:
        old["forest_raster"] = os.path.basename(
            str(old.get("forest_raster", "")).replace("\\", "/")
        )
        if old != sig:
            diffs = {k: (old.get(k), sig.get(k)) for k in sig if old.get(k) != sig.get(k)}
            raise RuntimeError(
                "This strip folder was created with different settings:\n"
                f"  (old, new) = {diffs}\n"
                "Use a new STRIP_OUTPUT_DIR, or restore the old settings."
            )
        return

    leftovers = strip_files_for_this_run() or all_checkpoint_paths()
    if leftovers and not RESUME_FROM_UNVERSIONED_RUN:
        raise RuntimeError(
            "The strip folder already contains strips from an earlier run that "
            "has no run_meta.json, so I can't verify they match these settings.\n"
            "  - Different area/settings: point STRIP_OUTPUT_DIR at a new folder.\n"
            "  - Resuming the SAME run with the SAME TEST_BBOX/STRIP_ROWS: set "
            "RESUME_FROM_UNVERSIONED_RUN = True."
        )

    tmp = meta_path.with_name(f"{meta_path.stem}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2)
    os.replace(tmp, meta_path)


# =============================================================================
# TASK LIST  (same windows/keys as the single-process version, so a run
# started earlier can be resumed)
# =============================================================================

def build_tasks(store, bbox, completed):
    tasks = []

    for zone, hemi in zone_groups_for_bbox(bbox):
        group = f"{zone:02d}{hemi}"
        zcrs = zone_crs(zone, hemi)

        print(f"\nPlanning zone group {group} ({zcrs})")

        try:
            ds = store.call(lambda: store.group(group), f"opening {group}")
        except Exception as exc:
            if is_missing_group(exc):
                print(f"  Group {group} not in store; skipped.")
                continue
            raise

        check_dataset(ds, group)
        print(f"  embeddings dims: {ds['embeddings'].dims}, shape: {ds['embeddings'].shape}")
        print(f"  embeddings chunks: {ds['embeddings'].encoding.get('chunks')}")

        t_idx = year_index(ds, YEAR)
        if t_idx is None:
            print(f"  Year {YEAR} not in {group}; skipped.")
            continue

        warn_if_year_incomplete(ds, group, YEAR)

        east = ds["easting"].values.astype("float64")
        north = ds["northing"].values.astype("float64")

        if east[1] < east[0]:
            raise ValueError("Easting is descending; unexpected layout.")

        north_ascending = bool(north[1] > north[0])
        res_x = float(abs(east[1] - east[0]))
        res_y = float(abs(north[1] - north[0]))

        xmin, ymin, xmax, ymax = transform_bounds(
            "EPSG:4326", zcrs, *bbox, densify_pts=21,
        )

        cols = np.where((east >= xmin - res_x) & (east <= xmax + res_x))[0]
        rows = np.where((north >= ymin - res_y) & (north <= ymax + res_y))[0]

        if len(cols) == 0 or len(rows) == 0:
            print("  Study area does not overlap this group; skipped.")
            continue

        c0, c1 = int(cols.min()), int(cols.max()) + 1
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        left = float(east[c0] - res_x / 2.0)

        n_strips = int(np.ceil((r1 - r0) / STRIP_ROWS))
        n_new = 0

        for k, s in enumerate(range(r0, r1, STRIP_ROWS), start=1):
            key = f"{group}_{k:04d}"
            if shard_of(key) != SHARD_INDEX:
                continue
            if key in completed:
                continue

            e = min(s + STRIP_ROWS, r1)
            top = float(north[e - 1] + res_y / 2.0) if north_ascending \
                else float(north[s] + res_y / 2.0)

            tasks.append({
                "key": key, "group": group, "zone": zone, "hemi": hemi,
                "crs": zcrs, "t_idx": t_idx,
                "s": s, "e": e, "c0": c0, "c1": c1,
                "left": left, "top": top,
                "res_x": res_x, "res_y": res_y,
                "north_ascending": north_ascending,
            })
            n_new += 1

        print(f"  Window rows {r0}-{r1}, cols {c0}-{c1}: "
              f"{n_strips} strips, {n_new} still to do")

    return tasks


# =============================================================================
# WORKERS
# =============================================================================

_W = {}


def init_worker(model_path, output_transform, output_width, output_height):
    """Runs once per worker process."""
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    _W.clear()
    try:
        bundle = joblib.load(model_path)
        model = bundle["model"]
        model.n_jobs = 1

        _W.update(
            model=model,
            predictor_cols=bundle["predictor_cols"],
            geo_indices=bundle["geo_indices"],
            store=TesseraStore(),
            soil={name: rasterio.open(p) for name, p in SOIL_RASTER_PATHS.items()},
            forest=rasterio.open(FOREST_RASTER),
            output_transform=output_transform,
            output_width=output_width,
            output_height=output_height,
        )
    except Exception as exc:
        # Never raise from a Pool initializer (the pool would respawn forever).
        _W["init_error"] = f"{type(exc).__name__}: {exc}"


def close_worker():
    for ds in _W.get("soil", {}).values():
        ds.close()
    if "forest" in _W:
        _W["forest"].close()


def run_task(task):
    """Process one strip. Never raises: returns (key, status)."""
    key = task["key"]

    if "init_error" in _W:
        return key, f"fatal: worker failed to start: {_W['init_error']}"

    try:
        F = AGGREGATION_FACTOR
        s, e, c0, c1 = task["s"], task["e"], task["c0"], task["c1"]
        height, width = e - s, c1 - c0
        agg_h, agg_w = height // F, width // F

        if agg_h == 0 or agg_w == 0:
            return key, "empty"

        crs = task["crs"]
        strip_transform = Affine(
            task["res_x"], 0.0, task["left"], 0.0, -task["res_y"], task["top"],
        )
        agg_transform = strip_transform * Affine.scale(F)

        agg = np.full((agg_h, agg_w), np.nan, dtype=np.float32)
        any_forest = False
        any_pred = False

        store = _W["store"]

        for cb in range(0, width, COL_BLOCK):
            cw = min(COL_BLOCK, width - cb)
            if cw < F:
                continue

            blk_transform = strip_transform * Affine.translation(cb, 0)
            blk_agg_transform = blk_transform * Affine.scale(F)

            # Skip blocks with no forest BEFORE downloading anything.
            if not strip_has_forest(
                _W["forest"], blk_agg_transform, crs, agg_h, cw // F,
            ):
                continue
            any_forest = True

            tile = store.call(
                lambda: read_strip(
                    store.group(task["group"]), task["t_idx"], _W["geo_indices"],
                    s, e, c0 + cb, c0 + cb + cw, task["north_ascending"],
                ),
                f"reading strip {key} (col block {cb})",
            )

            blk = predict_block(
                tile, blk_transform, crs,
                _W["model"], _W["predictor_cols"], _W["soil"],
            )
            del tile

            if blk is None:
                continue

            x0 = cb // F
            agg[:, x0:x0 + blk.shape[1]] = blk[:agg_h]
            any_pred = True
            del blk

        if not any_forest:
            return key, "noforest"
        if not any_pred:
            print(f"Strip {key}: no valid embeddings in forest blocks; nothing written.")
            return key, "empty"

        status = finish_strip(
            key, agg, agg_transform, crs, task["zone"], task["hemi"],
            _W["forest"], _W["output_transform"],
            _W["output_width"], _W["output_height"],
        )
        return key, status

    except Exception as exc:
        if is_fatal_error(exc):
            return key, f"fatal: {type(exc).__name__}: {exc}"
        import traceback
        traceback.print_exc()
        return key, f"error: {type(exc).__name__}: {exc}"

    finally:
        gc.collect()


# =============================================================================
# PARALLEL DRIVER
# =============================================================================

def predict_all(model_path, output_transform, output_width, output_height):
    print("\n" + "=" * 72)
    print("STEP 3 - PREDICT FROM AWS ICECHUNK")
    print("=" * 72)

    bbox = get_wgs84_bbox()
    check_run_meta(bbox)

    lock = acquire_run_lock()
    try:
        _predict_all_locked(
            bbox, model_path, output_transform, output_width, output_height,
        )
    finally:
        release_run_lock(lock)


def _predict_all_locked(bbox, model_path, output_transform, output_width, output_height):
    completed = existing_completed_strips()
    mine = load_json_keys(checkpoint_path())
    print(f"Shard {SHARD_INDEX + 1} of {NUM_SHARDS}")
    print(f"Existing completed strips (all shards): {len(completed)}")

    store = TesseraStore()
    tasks = build_tasks(store, bbox, completed)
    del store

    total = len(tasks)
    if total == 0:
        print("\nNothing left to do.")
        return

    print(f"\nStrips to process: {total:,}  |  workers: {WORKERS}")

    args = (str(model_path), output_transform, output_width, output_height)
    stats = {}
    failed = []
    t0 = time.time()
    done = 0
    ck = {"unsaved": 0, "t": time.time()}

    def record(key, status):
        nonlocal done
        done += 1

        if status.startswith("fatal"):
            raise RuntimeError(
                f"Stopping: {status}\n"
                "(Nothing bad was checkpointed; fix the problem and rerun to resume.)"
            )

        if status.startswith("error"):
            failed.append((key, status))
        else:
            completed.add(key)
            mine.add(key)
            stats[status] = stats.get(status, 0) + 1
            ck["unsaved"] += 1
            if ck["unsaved"] >= CHECKPOINT_EVERY or time.time() - ck["t"] >= 30:
                save_completed_strips(mine)
                ck["unsaved"], ck["t"] = 0, time.time()

        # Don't flood the console with thousands of instant no-forest skips.
        if status == "noforest" and done % 500 != 0:
            return

        elapsed = time.time() - t0
        eta = elapsed / done * (total - done)
        print(
            f"[{done:,}/{total:,}] {key}: {status} | "
            f"elapsed {elapsed / 3600:.2f} h, rough ETA {eta / 3600:.2f} h | "
            f"{stats}",
            flush=True,
        )

    try:
        if WORKERS <= 1:
            init_worker(*args)
            try:
                for task in tasks:
                    record(*run_task(task))
            finally:
                close_worker()
        else:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(
                processes=WORKERS, initializer=init_worker, initargs=args,
            ) as pool:
                for key, status in pool.imap_unordered(run_task, tasks, chunksize=1):
                    record(key, status)
    finally:
        save_completed_strips(mine)

    print("\nStrip processing finished.")
    print(f"Results: {stats}")

    if failed:
        print(f"\n{len(failed)} strips FAILED (not checkpointed). Re-run to retry them:")
        for key, status in failed[:20]:
            print(f"  {key}: {status}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


# =============================================================================
# OPTIONAL MOSAIC
# =============================================================================

def build_final_mosaic():
    print("\n" + "=" * 72)
    print("STEP 4 - BUILD FINAL MOSAIC")
    print("=" * 72)

    strip_files = strip_files_for_this_run()

    if not strip_files:
        print("No strip TIFFs found. No mosaic created.")
        return

    print(f"Found {len(strip_files):,} strip TIFFs.")

    profile, output_transform, width, height = create_output_grid()

    with rasterio.open(FINAL_MOSAIC, "w", **profile) as dst:

        blank_h = min(512, height)
        blank_w = min(512, width)
        blank = np.full((blank_h, blank_w), np.nan, dtype=np.float32)

        for row_off in range(0, height, blank_h):
            rows = min(blank_h, height - row_off)
            for col_off in range(0, width, blank_w):
                cols = min(blank_w, width - col_off)
                dst.write(
                    blank[:rows, :cols], 1,
                    window=Window(col_off, row_off, cols, rows),
                )
        del blank

    # Reopen read/write: a file opened with "w" cannot be read back.
    with rasterio.open(FINAL_MOSAIC, "r+") as dst:

        for i, strip_file in enumerate(strip_files, start=1):
            print(f"Mosaicking {i}/{len(strip_files)}: {strip_file.name}")

            with rasterio.open(strip_file) as src:
                window = output_window_from_bounds(
                    src.bounds, output_transform, width, height,
                )
                if window is None:
                    continue

                destination = np.full(
                    (int(window.height), int(window.width)),
                    np.nan, dtype=np.float32,
                )
                destination_transform = rasterio.windows.transform(
                    window, output_transform,
                )

                reproject(
                    source=rasterio.band(src, 1),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=destination_transform,
                    dst_crs=TARGET_CRS,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=Resampling.nearest,
                )

                existing = dst.read(1, window=window)
                valid = np.isfinite(destination)
                existing[valid] = destination[valid]
                dst.write(existing.astype(np.float32), 1, window=window)

                del destination, existing, valid

    print(f"\nFinal mosaic written to:\n{FINAL_MOSAIC}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    validate_paths()

    train_only = "--train-only" in sys.argv

    if USE_MODEL_CACHE and Path(MODEL_CACHE).exists():
        print(f"\nUsing cached model: {MODEL_CACHE}")
        print("(delete this file if the training data or settings change)")
        if train_only:
            print("Training only: model already exists, nothing to do.")
            return
    else:
        if NUM_SHARDS > 1 and not train_only:
            raise SystemExit(
                "No cached model yet. Train it ONCE before starting shards:\n"
                "  python tessera_rf_forest_strips.py --train-only\n"
                "then submit the array job."
            )
        model, predictor_cols, geo_indices = train_model()
        tmp = f"{MODEL_CACHE}.{os.getpid()}.tmp"
        joblib.dump(
            {
                "model": model,
                "predictor_cols": predictor_cols,
                "geo_indices": geo_indices,
            },
            tmp,
        )
        os.replace(tmp, MODEL_CACHE)
        print(f"Model saved to {MODEL_CACHE}")
        del model
        gc.collect()

    if train_only:
        print("Training only: done.")
        return

    (
        output_profile,
        output_transform,
        output_width,
        output_height,
    ) = create_output_grid()

    predict_all(
        model_path=MODEL_CACHE,
        output_transform=output_transform,
        output_width=output_width,
        output_height=output_height,
    )

    if BUILD_MOSAIC:
        build_final_mosaic()

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Individual strips:\n{STRIP_OUTPUT_DIR}")
    if BUILD_MOSAIC:
        print(f"Final mosaic:\n{FINAL_MOSAIC}")


if __name__ == "__main__":
    main()
