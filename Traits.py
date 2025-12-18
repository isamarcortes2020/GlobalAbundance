# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:31:56 2025

@author: cenv1124
"""

import polars as pl
import pickle
from pathlib import Path
from typing import Literal
import pyreadr
import glob

nfiles = sorted(glob.glob("R:/Global Dataset/US/AbundanceParquets/*.parquet"))
# ---------------------------
# Load LoadedData (single DF)
# ---------------------------
res1 = pyreadr.read_r("C:/Users/cenv1124/Downloads/TraitData.rds")
LoadedData = list(res1.values())[0]     # pandas DataFrame
traits_df = pl.from_pandas(LoadedData)  # Changed from pl.DataFrame()

n_files = len(nfiles)
df_list_up = [pl.read_parquet(f"R:/Global Dataset/US/AbundanceParquets/US_{i}.parquet")
              for i in range(1, n_files + 1)]


def assign_traits(
    individuals_df: pl.DataFrame,
    traits_df: pl.DataFrame,
    method: Literal["match_species", "any_species"] = "match_species",
    format: Literal["wide", "long"] = "wide"
) -> pl.DataFrame:
    """
    Assign trait values to individuals.
    
    Parameters:
    -----------
    individuals_df : pl.DataFrame
        DataFrame with 'species' column
    traits_df : pl.DataFrame
        DataFrame with 'species', 'trait_value', and 'trait' columns
    method : str
        'match_species' or 'any_species'
    format : str
        'wide' or 'long'
    """
    
    # Validate inputs
    if "species" not in individuals_df.columns:
        raise ValueError("individuals_df must contain 'species' column")
    
    required_cols = ["species", "trait_value", "trait"]
    if not all(col in traits_df.columns for col in required_cols):
        raise ValueError("traits_df must contain 'species', 'trait_value', and 'trait' columns")
    
    # Get unique traits - handle different return types
    try:
        unique_traits = traits_df["trait"].unique().to_list()
    except AttributeError:
        # Fallback for different polars versions or return types
        unique_traits = list(traits_df["trait"].unique())
    
    if method == "match_species":
        result = individuals_df.clone()
        
        # Add row index to track original rows
        result = result.with_row_index("_orig_idx")
        
        # For each trait, join and sample
        for trait_name in unique_traits:
            # Filter traits for this specific trait
            trait_subset = traits_df.filter(pl.col("trait") == trait_name)
            
            # Add row number for sampling within each species
            trait_subset = trait_subset.with_row_index("_trait_row")
            
            # Join individuals with their species' trait values
            joined = result.select(["_orig_idx", "species"]).join(
                trait_subset.select(["species", "trait_value", "_trait_row"]),
                on="species",
                how="left"
            )
            
            # Group by original row index and sample one trait value per individual
            sampled = (
                joined
                .group_by("_orig_idx", maintain_order=True)
                .agg(pl.col("trait_value").sample(n=1).first().alias(trait_name))
            )
            
            # Join the sampled values back to result
            result = result.join(
                sampled,
                on="_orig_idx",
                how="left"
            )
        
        # Drop the temporary index column
        result = result.drop("_orig_idx")
    
    elif method == "any_species":
        result = individuals_df.clone()
        n_individuals = len(individuals_df)
        
        # For each trait, randomly sample values
        for trait_name in unique_traits:
            trait_data = traits_df.filter(pl.col("trait") == trait_name)
            
            # Sample with replacement
            sampled_values = trait_data.select(
                pl.col("trait_value").sample(n=n_individuals, with_replacement=True)
            )
            
            result = result.with_columns(
                sampled_values.to_series().alias(trait_name)
            )
    
    else:
        raise ValueError("method must be either 'any_species' or 'match_species'")
    
    # Convert to long format if requested
    if format == "long":
        # Get non-trait columns
        id_cols = [col for col in result.columns if col not in unique_traits]
        
        # Melt to long format
        result = result.melt(
            id_vars=id_cols,
            value_vars=unique_traits,
            variable_name="trait",
            value_name="trait_value"
        )
    
    return result



def assign_traits_to_list(
    df_list: list[pl.DataFrame],
    traits_df: pl.DataFrame,
    method: Literal["match_species", "any_species"] = "match_species",
    format: Literal["wide", "long"] = "wide",
    output_dir: str = "R:/Global Dataset/US/WithTraits"
) -> list[pl.DataFrame]:
    """
    Apply trait assignment to a list of DataFrames and save results.
    """
    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, df in enumerate(df_list):
        result_df = assign_traits(
            individuals_df=df,
            traits_df=traits_df,
            method=method,
            format=format
        )
        
        # Save as parquet (more efficient than pickle for DataFrames)
        result_df.write_parquet(output_path / f"US_{i}.parquet")
        
        results.append(result_df)
    
    return results

# Apply trait assignment to all dataframes in the list
result_list = assign_traits_to_list(
    df_list=df_list_up,
    traits_df=traits_df,
    method="match_species",
    format="wide"
)

