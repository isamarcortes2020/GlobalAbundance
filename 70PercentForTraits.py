import pickle
import glob

trait = "Leaf N content (%)"
all_filtered_dfs = []

# Get all pickle files in a folder
pkl_files = glob.glob('R:/Global Dataset/70PercentCoverage/Chunk1/*.pkl')

for file in pkl_files:
    with open(file, "rb") as f:
        data = pickle.load(f)☺

    # assuming list of lists of DataFrames
    for sublist in data:
        for df in sublist:
            if trait in df.columns:
                coverage = df[trait].notna().mean()
                if coverage > 0.7:
                    all_filtered_dfs.append(df)

print(f"Total kept: {len(all_filtered_dfs)}")


output_file = "R:/Global Dataset/CoveragePerTrait/LeafNContent/LeafNContent1.pkl"

with open(output_file, "wb") as f:
    pickle.dump(all_filtered_dfs, f)

print(f"Saved filtered data to: {output_file}")
