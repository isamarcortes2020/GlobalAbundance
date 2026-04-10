import pandas as pd
import numpy as np
import glob
from geotessera import GeoTessera
import os
from tqdm import tqdm

# ----------------------------
# Setup
# ----------------------------
os.environ["GEOTESSERA_CACHE_DIR"] = "/soge-home/users/cenv1124/scratch/geotessera_cache"

Files = sorted(glob.glob("/soge-home/users/cenv1124/CWMTraits/BranchHydraulic/*.csv"))

# ----------------------------
# Load + filter data
# ----------------------------
Data = []
for f in Files:
    data = pd.read_csv(f)
    data = data[(data["Year"] >= 2010) & (data["Year"] <= 2016)]
    Data.append(data)

Data = pd.concat(Data)

# Clean coords
Data['Coords_x'] = Data['Coords_x'].astype(float)
Data['Coords_y'] = Data['Coords_y'].astype(float)

# Trait filter
trait = "Branch hydraulic conductance kg.m-1.MPa-1.s-1"
df = Data.dropna(subset=[trait]).reset_index(drop=True)

# ----------------------------
# GeoTessera
# ----------------------------
gt = GeoTessera(cache_dir="/soge-home/users/cenv1124/scratch/geotessera_cache")

# ----------------------------
# Sequential embedding
# ----------------------------
all_embeddings = []

chunk_size = 500  # avoid memory issues

for i in tqdm(range(0, len(df), chunk_size), desc="Embedding points"):
    chunk = df.iloc[i:i+chunk_size]

    points = list(zip(chunk['Coords_x'], chunk['Coords_y']))

    emb = gt.sample_embeddings_at_points(points, year=2017)

    emb_df = pd.DataFrame(emb, index=chunk.index)

    all_embeddings.append(emb_df)

# ----------------------------
# Combine safely
# ----------------------------
embeddings_df = pd.concat(all_embeddings).sort_index()

# join (safe because indices match)
df = df.join(embeddings_df)

# ----------------------------
# Save
# ----------------------------
output_path = "/soge-home/users/cenv1124/CWMTraits/DataCombinedWithTessera/BranchHydraulic2010_2016.csv"
df.to_csv(output_path, index=False)

print("Done ✅")

