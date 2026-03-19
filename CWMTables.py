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
  
  



File = 'R:/GolbalDataset/70PercentCoverage/Canada8.pkl'
test = CWM(File)
test.to_csv('R:/GolbalDataset/CWM/Canada8.csv')
#combined_df = pd.concat(test, ignore_index=True)    
#combined_df.columns = combined_df.columns.str.replace('/', '_')

