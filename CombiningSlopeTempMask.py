# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:23:21 2026

@author: cenv1124
"""

import pandas as pd


Data = pd.read_csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/UpdatedDatasetSoFar/LeafArea/LeafAreaCombined.csv")
SRTM = pd.read_csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/UpdatedDatasetSoFar/LeafArea/LeafArea_SRTM.csv")
Tmax = pd.read_csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/UpdatedDatasetSoFar/LeafArea/LeafAreaTmax_Mean_MCWD_1988_2017.csv")
Mask = pd.read_csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/UpdatedDatasetSoFar/LeafArea/LeafArea_DynamicWorld_Class_Extraction.csv")


# Merge step by step
merged = Data.merge(
    SRTM[['ID', 'slope']],
    on='ID',
    how='left'
)

merged = merged.merge(
    Tmax[['ID', 'mcwd', 'tmax_mean']],
    on='ID',
    how='left'
)

merged = merged.merge(
    Mask[['ID', 'label']],
    on='ID',
    how='left'
)

# Save
merged.to_csv(
    r"R:/GlobalDataset/TraitsCombinedWithGeoTessera/FinalizedTesseraData/LeafAreaCombined.csv",
    index=False
)
