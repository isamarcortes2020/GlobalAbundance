library('ggplot2')
library('readxl')
library('dplyr')
library('readr')
library('readxl')
library('tidyr')
library('arrow')

df_list_up <- readRDS("L:/Canada/abundanceCanada.rds")


# Check if it's a dataframe/tibble
if (is.data.frame(df_list_up)) {
  # Split into 20 roughly equal chunks
  n_chunks <- 20
  n_rows <- nrow(df_list_up)
  chunk_size <- ceiling(n_rows / n_chunks)
  
  # Create and save each chunk
  for (i in 1:n_chunks) {
    start_row <- (i - 1) * chunk_size + 1
    end_row <- min(i * chunk_size, n_rows)
    
    chunk <- df_list_up[start_row:end_row, ]
    
    # Save chunk
    saveRDS(chunk, file = paste0("L:/Canada/test/Canada_chunk_", i, ".rds"))
    cat("Saved chunk", i, "with", nrow(chunk), "rows\n")
  }
  
} else if (is.list(df_list_up)) {
  # If it's a list, split the list into chunks
  n_chunks <- 20
  chunk_size <- ceiling(length(df_list_up) / n_chunks)
  
  for (i in 1:n_chunks) {
    start_idx <- (i - 1) * chunk_size + 1
    end_idx <- min(i * chunk_size, length(df_list_up))
    
    chunk <- df_list_up[start_idx:end_idx]
    
    saveRDS(chunk, file = paste0("L:/Canada/test/Canada_chunk_", i, ".rds"))
    cat("Saved chunk", i, "with", length(chunk), "elements\n")
  }
}
