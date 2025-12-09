# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:57:30 2025

@author: cenv1124
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:17:31 2025

@author: cenv1124
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pickle

Files = sorted(glob.glob("R:/Global Dataset/Mexico/withTraits/*.parquet"))

Dataset = []



for i in Files:
    test = pd.read_parquet(i)
    Dataset.append(test)
    

t = Dataset[1]
names = t.columns[7:].tolist()




MoreThan70 = []
for i in names:

    filtered_dfs = [
        df for df in Dataset
        if df[i].notna().mean() >= 0.7
    ]
    MoreThan70.append(filtered_dfs)

cleaned1 = [lst for lst in MoreThan70 if lst]


seen = set()
cleaned = []

for group in cleaned1:         # outer list
    new_group = []
    for df in group:       # inner list of dataframes
        
        # Make sure df is actually a DataFrame
        if hasattr(df, "columns") and "PlotID" in df.columns:
            plot_id = df["PlotID"].unique()[0]
        else:
            continue  # skip if it's not a dataframe
        
        if plot_id not in seen:
            new_group.append(df)
            seen.add(plot_id)
    
    # Keep inner lists that are not empty
    if new_group:
        cleaned.append(new_group)
        
        
        
'''
####To check duplicate PlotID dataframes
seen = {}
duplicates = {}      # <-- NEW: store full dataframes for repeated PlotIDs
cleaned = []

for group in cleaned1:         # outer list
    new_group = []
    
    for df in group:
        # Ensure df is a DataFrame with PlotID
        if not hasattr(df, "columns") or "PlotID" not in df.columns:
            continue
        
        plot_id = df["PlotID"].unique()[0]

        # If this PlotID was seen before → it's a duplicate
        if plot_id in seen:
            # Add this df to the duplicates dictionary
            duplicates.setdefault(plot_id, []).append(df)
            continue
        
        # First time seeing this PlotID → keep it
        new_group.append(df)
        seen[plot_id] = df   # store first appearance
    
    if new_group:
        cleaned.append(new_group)

'''


with open("R:/Global Dataset/CWMat70PercentCov/Mexico.pkl", "wb") as f:
    pickle.dump(cleaned, f)


