import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

#test = sorted(glob.glob('C:/Users/cenv1124/Downloads/Mexico/withTraits/*.parquet'))
#nfiles = len(test)

def compute_cwm_from_files(base_path, n_files=10000):
    """
    Simple function to compute CWM across multiple parquet files.
    
    Args:
        base_path: Path like "C:/Users/cenv1124/Downloads/Mexico/withTraits/Mexico"
        n_files: Number of files (default 19)
    
    Returns:
        pandas DataFrame with CWM for each file
    """
    metadata_cols = ("species", "B", "Ps", "Coords_y", "Coords_x", "Yr", "PlotID")
    cwm_list = []
    
    for i in range(1, n_files + 1):
        # Read file
        df = pl.read_parquet(f"{base_path}_{i}.parquet").to_pandas()
        
        # Get numeric trait columns
        trait_cols = [
            col for col in df.columns
            if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        # Group by species
        grouped = df.groupby("species")
        n_ind = grouped.size()
        trait_means = grouped[trait_cols].mean()
        total_n = n_ind.sum()
        
        # Compute CWMs
        cwms = {}
        for col in trait_cols:
            cwms[col] = np.nansum(n_ind * trait_means[col]) / total_n
        
        cwm_list.append(pd.DataFrame([cwms]))
    
    return pd.concat(cwm_list, ignore_index=True)


# Usage - just one line!
cwm_table = compute_cwm_from_files("C:/Users/cenv1124/Downloads/Processed_Parquets-20251202T113024Z-1-001/Processed_Parquets/Africa", n_files=10000)
    



# Assuming you have cwm_table from the previous function
# cwm_table = compute_cwm_from_files("path", n_files=19)

def visualize_cwm_trends(cwm_table):
    """Line plots showing how each trait varies across files/plots."""
    trait_cols = [col for col in cwm_table.columns if col not in ['file_name', 'file_index']]
    
    n_traits = len(trait_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_traits / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_traits > 1 else [axes]
    
    for i, trait in enumerate(trait_cols):
        ax = axes[i]
        ax.plot(range(len(cwm_table)), cwm_table[trait], marker='o', linewidth=2)
        ax.set_xlabel('File Index', fontsize=10)
        ax.set_ylabel(f'CWM {trait}', fontsize=10)
        ax.set_title(f'{trait}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_traits, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('cwm_trends.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_cwm_heatmap(cwm_table):
    """Heatmap showing all traits across all files."""
    trait_cols = [col for col in cwm_table.columns if col not in ['file_name', 'file_index']]
    
    # Normalize each trait to 0-1 scale for better comparison
    cwm_normalized = cwm_table[trait_cols].copy()
    for col in trait_cols:
        min_val = cwm_normalized[col].min()
        max_val = cwm_normalized[col].max()
        cwm_normalized[col] = (cwm_normalized[col] - min_val) / (max_val - min_val)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cwm_normalized.T, 
                cmap='viridis', 
                cbar_kws={'label': 'Normalized CWM'},
                xticklabels=range(1, len(cwm_table)+1),
                yticklabels=trait_cols)
    plt.xlabel('File Index', fontsize=12)
    plt.ylabel('Trait', fontsize=12)
    plt.title('Community Weighted Means Heatmap (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cwm_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_cwm_correlation(cwm_table):
    """Correlation matrix between traits."""
    trait_cols = [col for col in cwm_table.columns if col not in ['file_name', 'file_index']]
    
    corr = cwm_table[trait_cols].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr), k=1)  # Mask upper triangle
    sns.heatmap(corr, 
                mask=mask,
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={'label': 'Correlation'})
    plt.title('Trait Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cwm_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_cwm_distributions(cwm_table):
    """Box plots showing distribution of each trait's CWM across files."""
    trait_cols = [col for col in cwm_table.columns if col not in ['file_name', 'file_index']]
    
    n_traits = len(trait_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_traits / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_traits > 1 else [axes]
    
    for i, trait in enumerate(trait_cols):
        ax = axes[i]
        ax.boxplot([cwm_table[trait]], labels=[trait])
        ax.scatter([1]*len(cwm_table), cwm_table[trait], alpha=0.5, s=50)
        ax.set_ylabel('CWM Value', fontsize=10)
        ax.set_title(f'{trait}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide empty subplots
    for i in range(n_traits, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('cwm_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_cwm_pca(cwm_table):
    """PCA biplot to see patterns across files and traits."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    trait_cols = [col for col in cwm_table.columns if col not in ['file_name', 'file_index']]
    
    # Standardize the data
    scaler = StandardScaler()
    cwm_scaled = scaler.fit_transform(cwm_table[trait_cols])
    
    # PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(cwm_scaled)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot of files
    scatter = ax1.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                          c=range(len(cwm_table)), cmap='viridis', 
                          s=100, alpha=0.7, edgecolors='black')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax1.set_title('Files in PCA Space', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='File Index')
    
    # Loading plot
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, trait in enumerate(trait_cols):
        ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                  head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
        ax2.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, trait, 
                fontsize=9, ha='center', va='center')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax2.set_title('Trait', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('cwm_pca.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_all_visualizations(cwm_table):
    """Generate all visualizations at once."""
    print("Creating trend plots...")
    visualize_cwm_trends(cwm_table)
    
    print("Creating heatmap...")
    visualize_cwm_heatmap(cwm_table)
    
    print("Creating correlation matrix...")
    visualize_cwm_correlation(cwm_table)
    
    print("Creating distribution plots...")
    visualize_cwm_distributions(cwm_table)
    
    print("Creating PCA biplot...")
    visualize_cwm_pca(cwm_table)
    
    print("All visualizations saved!")



visualize_cwm_heatmap(cwm_table)


# Usage:
# create_all_visualizations(cwm_table)

# Or individual plots:
# visualize_cwm_trends(cwm_table)
# visualize_cwm_heatmap(cwm_table)
# visualize_cwm_correlation(cwm_table)
# visualize_cwm_distributions(cwm_table)
# visualize_cwm_pca(cwm_table)
