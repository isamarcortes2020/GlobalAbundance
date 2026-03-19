import pandas as pd
import glob
import os

# Folder containing your CSV files
folder_path = "R:/GolbalDataset/CWM/*.csv"

all_files = glob.glob(folder_path)

summary_list = []

for file in all_files:
    df = pd.read_csv(file)
    
    # Count non-NA values per column
    counts = df.count()
    
    # Convert to dataframe
    counts_df = counts.to_frame(name="count").T
    
    # Add filename
    counts_df["file_name"] = os.path.basename(file)
    
    summary_list.append(counts_df)

# Combine all into one dataframe
master_df = pd.concat(summary_list, ignore_index=True)

# Set filename as index (optional but useful for heatmaps)
master_df = master_df.set_index("file_name")

master_df = master_df.drop(columns=["Coords_x", "Coords_y", "PlotID","Year","DBH_cm"])

print(master_df.head())



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

sns.heatmap(master_df, cmap="viridis")

plt.title("Number of Observations per Trait per File")
plt.xlabel("Traits")
plt.ylabel("CSV Files")

plt.tight_layout()
plt.show()
