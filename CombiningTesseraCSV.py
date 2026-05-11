import pandas as pd
import glob
import os

# Folder containing CSV files
folder = r"R:/GlobalDataset/TraitsCombinedWithGeoTessera/UpdatedDatasetSoFar/RootDryMass"

# Get all CSV files in folder
csv_files = glob.glob(os.path.join(folder, "*.csv"))

# Read all CSVs into a list
dfs = [pd.read_csv(file) for file in csv_files]

# Find common columns while PRESERVING order from first dataframe
common_cols = dfs[0].columns

for df in dfs[1:]:
    common_cols = [col for col in common_cols if col in df.columns]

# Combine all dataframes using only common columns
result = pd.concat(
    [df[common_cols] for df in dfs],
    axis=0,
    ignore_index=True
)

# Create ID column
result["ID"] = range(len(result))

# Save combined CSV
output_file = r"R:/GlobalDataset/TraitsCombinedWithGeoTessera/UpdatedDatasetSoFar/RootDryMass/RootDryMassCombined.csv"

result.to_csv(output_file, index=False)

print(f"Combined {len(csv_files)} files")
print(f"Saved to: {output_file}")
