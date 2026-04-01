import pickle
from itertools import chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os


def CWM(PKLFile):
    with open(PKLFile, "rb") as f:
        data = pickle.load(f)

    CWM_mainTable = []

    for t in data:
        # if t is a DataFrame → wrap it in a list
        if isinstance(t, pd.DataFrame):
            t = [t]

        for s in t:
            Traits = list(s.columns[7:])
            CWM = s[Traits].mean()

            CWM['Coords_y'] = s['Coords_y'].iloc[0]
            CWM['Coords_x'] = s['Coords_x'].iloc[0]
            CWM['PlotID'] = s['PlotID'].iloc[0]
            CWM['Year'] = s['Yr'].iloc[0]

            CWM_mainTable.append(CWM)

    df = pd.DataFrame(CWM_mainTable)
    df['Coords_x'] = pd.to_numeric(df['Coords_x'], errors='coerce')
    df['Coords_y'] = pd.to_numeric(df['Coords_y'], errors='coerce')

    return df




# Folder containing pickle files
input_folder = r"R:/GolbalDataset/CoveragePerTrait/LeafFreshMass/"
output_folder = r'R:/GolbalDataset/CWMTraits/'

# Get all .pkl files
pkl_files = glob.glob(os.path.join(input_folder, '*.pkl'))

for file in pkl_files:
    print(f"Processing: {file}")

    df = CWM(file)

    # Create output filename with same base name
    base_name = os.path.splitext(os.path.basename(file))[0]
    output_file = os.path.join(output_folder, base_name + '.csv')

    df.to_csv(output_file, index=False)

print("All files processed.")
