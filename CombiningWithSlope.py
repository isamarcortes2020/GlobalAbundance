# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:51:13 2026

@author: cenv1124
"""

import pandas as pd

slope = pd.read_csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/UpdatedDatasetSoFar/RootDryMass/RootDryMass_SRTM.csv")
data = pd.read_csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/FinalizedTesseraData/RootDryMassCombined.csv")


merged = pd.merge(
    data,
    slope[['ID', 'slope']],
    on='ID',
    how='left'
)

merged.to_csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/FinalizedTesseraData/RootDryMassCombined.csv")
