# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:31:51 2026

@author: cenv1124
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re


long_df = pd.read_csv("R:/GolbalDataset/MapFig.csv")


long_df = long_df[long_df["trait"] != "Unnamed: 0"]

traits = long_df["trait"].unique()


sns.jointplot(data=long_df,x="Coords_x",y="Coords_y")

for trait in traits:
    subset = long_df[long_df["trait"] == trait]
    
    g = sns.jointplot(
        data=subset,
        x="Coords_x",
        y="Coords_y"
    )
    
    g.fig.suptitle(trait)
    safe_trait = re.sub(r"[^\w]+", "_", trait).strip("_")
    # Save figure
    g.fig.savefig(f"R:/GolbalDataset/TraitMaps/{safe_trait}.png")
    plt.close()



