library(ggplot2)
library(dplyr)
library(readr)
library(readxl)
library(tidyr)
library(future)
library(future.apply)

# ---- Parallel setup ----
plan(multicore, workers = 10)  # adjust to your available cores
options(future.globals.maxSize = 8 * 1024^3)  # 8 GiB for chunked objects

# ---- CSV chunk processing ----
process_chunk <- function(df, pos){
  df %>%
    filter(Yr >= 1975)
}



Afr <- '/soge-home/users/cenv1124/NorthAmerica_RestOfArea_abM_v4_7.csv'

df11 <- read_csv_chunked(
  Afr,
  callback = DataFrameCallback$new(process_chunk),
  chunk_size = 100000
)

S_Amer <- na.omit(df11)
S_Amer_test <- t(S_Amer)
S_Amer_test <- cbind(newColName = rownames(S_Amer_test), S_Amer_test)
S_Amer_test <- as.data.frame(S_Amer_test)
S_Amer_test_firstcol <- S_Amer_test$newColName

# ---- Process columns locally (huge object, do not parallelize) ----
column_dfs <- lapply(names(S_Amer_test), function(col_name) {
  data.frame(S_Amer_test[[col_name]])
})

df_list <- lapply(column_dfs, function(df) {
  df$col <- S_Amer_test_firstcol
  df$PlotID <- df[1,1]
  df
})

# ---- Clean and expand locally ----
df_list_cleaned <- lapply(df_list[-1], function(df){
  df <- na.omit(df)
  df[apply(df != 0, 1, all), ]
})

expanded_list <- vector("list", length(df_list_cleaned))

for (i in seq_along(df_list_cleaned)) {
  df <- df_list_cleaned[[i]]
  
  test1 <- df[1:7, , drop = FALSE]
  test1 <- as.data.frame(t(test1))
  rowNames <- c("PlotID","Place","Ps","Coords_y","Coords_x","B","Yr")
  colnames(test1) <- rowNames
  test1 <- test1[-2, ]
  test1[is.na(test1)] <- 0
  test1 <- test1[-2, ]
  
  test2 <- df[8:nrow(df), , drop = FALSE]
  test2 <- na.omit(test2)
  
  expanded_rest <- data.frame(
    species = rep(test2[['col']], test2[['S_Amer_test..col_name..']])
  )
  
  expanded_rest$B <- rep(test1$B, nrow(expanded_rest))
  expanded_rest$Ps <- rep(test1$Ps, nrow(expanded_rest))
  expanded_rest$Coords_y <- rep(test1$Coords_y, nrow(expanded_rest))
  expanded_rest$Coords_x <- rep(test1$Coords_x, nrow(expanded_rest))
  expanded_rest$Yr <- rep(test1$Yr, nrow(expanded_rest))
  expanded_rest$PlotID <- rep(test1$PlotID, nrow(expanded_rest))
  
  expanded_list[[i]] <- expanded_rest
}

# ---- Parallelize smaller chunks ----
chunk_size <- 50
chunk_indices <- split(seq_along(expanded_list), ceiling(seq_along(expanded_list)/chunk_size))

df_list_up <- future_lapply(chunk_indices, function(idxs) {
  lapply(idxs, function(i) {
    df <- expanded_list[[i]]
    df$species <- gsub("\\.", " ", df$species)
    df
  })
}) |> unlist(recursive = FALSE)
