library('readr')
library('readxl')
library('tidyr')
library('arrow')

df_list_up <- readRDS("L:/US/abundanceUS.rds")


# Function to convert a dataframe's character columns to UTF-8
convert_to_utf8 <- function(df) {
  df[] <- lapply(df, function(col) {
    if (is.character(col)) {
      # Convert to UTF-8 safely
      Encoding(col) <- "unknown"  # reset any existing encoding
      col <- iconv(col, from = "", to = "UTF-8", sub = "byte")  # convert, replacing invalid bytes
    }
    col
  })
  df
}

# Apply to each dataframe in the list
df_list_up_utf8 <- lapply(df_list_up, convert_to_utf8)

# Save as Parquet (recommended) or RDS
library(arrow)
for (i in seq_along(df_list_up)) {
  write_parquet(df_list_up[[i]], paste0("R:/Global Dataset/US/AbundanceParquets/US_", i, ".parquet"))
}
