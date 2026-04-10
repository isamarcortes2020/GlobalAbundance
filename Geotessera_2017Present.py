# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:59:08 2026

@author: cenv1124
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:53:52 2026

@author: cenv1124
"""

import pandas as pd
import numpy as np
import glob
from geotessera import GeoTessera
import json
from sklearn.model_selection import train_test_split
import os
os.environ["GEOTESSERA_CACHE_DIR"] = "/soge-home/users/cenv1124/scratch/geotessera_cache"


Files = sorted(glob.glob("/soge-home/users/cenv1124/CWMTraits/StemDryMass/*.csv"))

Data = []

for i in Files:
    data = pd.read_csv(i)
    data = data[data["Year"] >= 2017]
    Data.append(data)


Data = pd.concat(Data)
Data['Coords_x']=Data['Coords_x'].astype(float)
Data['Coords_y']=Data['Coords_y'].astype(float)

gt = GeoTessera(cache_dir="/soge-home/users/cenv1124/scratch/geotessera_cache")

trait = "Stem dry mass (kg)" ######change this
df = Data.dropna(subset=[trait]).reset_index(drop=True)
coords = df[["Coords_x", "Coords_y"]].values
points = list(zip(df['Coords_x'], df['Coords_y']))


from tqdm import tqdm

all_embeddings = []

groups = list(df.groupby("Year"))

# total number of points
total_points = len(df)

with tqdm(total=total_points, desc="Processing points") as pbar:
    
    for year, group in groups:
        points = list(zip(group['Coords_x'], group['Coords_y']))
        
        emb = gt.sample_embeddings_at_points(points, year=year)
        
        emb_df = pd.DataFrame(emb, index=group.index)
        all_embeddings.append(emb_df)
        
        # update progress by number of points processed
        pbar.update(len(group))

# combine results
embeddings_df = pd.concat(all_embeddings).sort_index()
df = pd.concat([df, embeddings_df], axis=1)
df.to_csv('/soge-home/users/cenv1124/CWMTraits/DataCombinedWithTessera/StemDryMass2017_Present.csv')


'''
for year, group in df.groupby("Year"):
    #points = group[["Coords_x", "Coords_y"]].values
    emb = gt.sample_embeddings_at_points(points, year=year)
    #all_embeddings.append(pd.DataFrame(emb, index=group.index))

'''

