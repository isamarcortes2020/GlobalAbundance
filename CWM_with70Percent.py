import pickle
from itertools import chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob


def CWM(PKLFile):
    with open(PKLFile,"rb") as f:
        data = pickle.load(f)

    CWM_mainTable = []
    for i in range(len(data)):
        t = data[i]
        for j in range(len(t)):
            s = t[j]
            Traits = list(s.columns[7:])
            CWM = s[Traits].mean()
            #CWM = (CWM - CWM.min()) / (CWM.max() - CWM.min())
            CWM['Coords_y'] = s['Coords_y'].iloc[0]
            CWM['Coords_x'] = s['Coords_x'].iloc[0]
            CWM['PlotID'] = s['PlotID'].iloc[0]
            CWM['Year']=s['Yr'].iloc[0]
            CWM_mainTable.append(CWM)     
    df = pd.DataFrame(CWM_mainTable)
    df['Coords_x'] = pd.to_numeric(df['Coords_x'], errors='coerce')
    df['Coords_y'] = pd.to_numeric(df['Coords_y'], errors='coerce')
    return df
    
"R:/GolbalDataset/CanadaMaps"
File = sorted(glob.glob("R:/GolbalDataset/CanadaMaps/*.pkl"))
#File = "R:/GolbalDataset/US2_Maps/US4.pkl"

data = []
for i in File:
    test = CWM(i)
    data.append(test)
    
combined_df = pd.concat(data, ignore_index=True)    
combined_df.columns = combined_df.columns.str.replace('/', '_')





'''

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


Traits = list(combined_df.columns[:47])


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

Traits = list(combined_df.columns[:47])  # your trait columns
output_dir = "R:/GolbalDataset/US3_Maps/"    # change this to your folder
os.makedirs(output_dir, exist_ok=True)

for trait in Traits:
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Scatter plot
    scatter = ax.scatter(
        combined_df["Coords_x"], 
        combined_df["Coords_y"], 
        c=combined_df[trait], 
        cmap="viridis", 
        s=10, 
        transform=ccrs.PlateCarree()
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.7)
    cbar.set_label(f"CWM {trait}")

    # Title
    ax.set_title(f"CWM map: {trait}")

    # Optional: zoom to your region (Europe example)
    #ax.set_extent([5, 15, 45, 55], crs=ccrs.PlateCarree())

    # Save figure
    plt.savefig(os.path.join(output_dir, f"CWM_map_{trait}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)  # close figure to save memory
'''
