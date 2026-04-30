# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:17:48 2026

@author: cenv1124
"""


import numpy as np
import matplotlib.pyplot as plt
from geotessera import GeoTessera
import json
import pandas as pd



Amax = pd.read_csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/Amax.csv")

df = Amax.dropna(subset=["0"])
embeddings_only = df.loc[:, "0":]


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Define features (columns 0-127 from TESSERA) and target trait
X = embeddings_only
y = df['Amax']

# Split for validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestRegressor(
    n_estimators=800,
    max_depth=None,              # remove depth cap
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("R2:", r2)
print("RMSE:", rmse)
print("MAE:", mae)


from sklearn.model_selection import cross_val_score, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

print("R2 scores:", r2_scores)
print("Mean R2:", r2_scores.mean())



