library(dplyr)
library(sf)
library(rnaturalearth)
library(ggplot2)

# Trait labels lookup (folder name -> pretty label)
trait_labels <- c(
  "Amax"                     = "Amax",
  "Asat"                     = "Asat",
  "BranchHydraulic"          = "Branch hydraulic conductance kg.m-1.MPa-1.s-1",
  "FruitLength"              = "Fruit length (mm)",
  "LeafArea"                 = "Leaf area (cm2)",
  "LeafCaContent"            = "Leaf Ca content (%)",
  "LeafCcontent"             = "Leaf C content (%)",
  "LeafDryMass"              = "Leaf dry mass (g)",
  "LeafFreshMass"            = "Leaf fresh mass (g)",
  "LeafHydraulic"            = "Leaf hydraulic conductance mmol m-2 s-1 MPa-1",
  "LeafKContent"             = "Leaf K content (%)",
  "LeafMgContent"            = "Leaf Mg content (%)",
  "LeafNContent"             = "Leaf N content (%)",
  "LeafPContent"             = "Leaf P content (%)",
  "LeafThickness"            = "Leaf thickness (mm)",
  "LeafWaterContent"         = "Leaf water content",
  "RootDryMass"              = "Root dry mass (g)",
  "SeedLength"               = "Seed length (mm)",
  "SeedMass"                 = "Seed mass (mg)",
  "SLA"                      = "SLA (cm2/g)",
  "StemDryMass"              = "Stem dry mass (kg)",
  "STEMHydraulicConductance" = "Stem hydraulic conductivity kg s-1 MPa-1",
  "VesselLumen"              = "Vessel lumen area (mm2)",
  "VesselNumber"             = "Vessel number (mm2)",
  "WoodDensity"              = "Wood density (g/cm3)"
)

# Folder name -> exact column name in CSV (as read.csv converts them with dots)
trait_cols <- c(
  "Amax"                     = "Amax",
  "Asat"                     = "Asat",
  "BranchHydraulic"          = "Branch.hydraulic.conductance.kg.m.1.MPa.1.s.1",
  "FruitLength"              = "Fruit.length..mm.",
  "LeafArea"                 = "Leaf.area..cm2.",
  "LeafCaContent"            = "Leaf.Ca.content....",
  "LeafCcontent"             = "Leaf.C.content....",
  "LeafDryMass"              = "Leaf.dry.mass..g.",
  "LeafFreshMass"            = "Leaf.fresh.mass..g.",
  "LeafHydraulic"            = "Leaf.hydraulic.conductance.mmol.m.2.s.1.MPa.1",
  "LeafKContent"             = "Leaf.K.content....",
  "LeafMgContent"            = "Leaf.Mg.content....",
  "LeafNContent"             = "Leaf.N.content....",
  "LeafPContent"             = "Leaf.P.content....",
  "LeafThickness"            = "Leaf.thickness..mm.",
  "LeafWaterContent"         = "Leaf_Water_Content",
  "RootDryMass"              = "Root.dry.mass..g.",
  "SeedLength"               = "Seed.length..mm.",
  "SeedMass"                 = "Seed.mass..mg.",
  "SLA"                      = "SLA..cm2.g.",
  "StemDryMass"              = "Stem.dry.mass..kg.",
  "STEMHydraulicConductance" = "Stem.hydraulic.conductivity.kg.s.1.MPa.1",
  "VesselLumen"              = "Vessel.lumen.area..mm2.",
  "VesselNumber"             = "Vessel.number..mm2.",
  "WoodDensity"              = "Wood.density..g.cm3."
)

# Base directory containing trait folders
base_dir <- "R:/Global Dataset/CWMTraits"
trait_dirs <- list.dirs(base_dir, recursive = FALSE)

# World map
world <- ne_countries(scale = "medium", returnclass = "sf")

for (trait_path in trait_dirs) {
  
  trait_name <- basename(trait_path)
  trait_label <- ifelse(!is.na(trait_labels[trait_name]), trait_labels[trait_name], trait_name)
  trait_col   <- trait_cols[trait_name]
  
  if (is.na(trait_col)) {
    message("No column mapping for: ", trait_name, " — skipping")
    next
  }
  
  # Read CSV files
  files <- list.files(path = trait_path, pattern = "\\.csv$", full.names = TRUE)
  
  if (length(files) == 0) {
    message("No CSV files found in: ", trait_name, " — skipping")
    next
  }
  
  combined_df <- do.call(rbind, lapply(files, read.csv))
  
  if (is.null(combined_df) || nrow(combined_df) == 0) {
    message("Empty data for: ", trait_name, " — skipping")
    next
  }
  
  # Check the trait column actually exists
  if (!trait_col %in% names(combined_df)) {
    message("Column '", trait_col, "' not found in: ", trait_name, " — skipping")
    message("  Available columns: ", paste(names(combined_df), collapse = ", "))
    next
  }
  
  # Filter years
  filtered_df <- combined_df %>%
    filter(Year >= 2010)
  
  if (nrow(filtered_df) == 0) {
    message("No data after year filter for: ", trait_name, " — skipping")
    next
  }
  
  # Create plot
  p <- ggplot(data = world) +
    geom_sf(fill = "grey95", color = "grey70") +
    geom_point(
      data = filtered_df,
      aes_string(x = "Coords_x", y = "Coords_y", color = trait_col),
      size = 1,
      alpha = 0.6
    ) +
    theme_minimal() +
    xlab("") + ylab("") +
    labs(
      title = paste("CWM", trait_label, "2010 - 2025"),
      color = trait_label
    )
  
  # Save plot
  ggsave(
    filename = paste0("R:/Global Dataset/CWM_TraitMaps/Map_", trait_name, ".png"),
    plot = p,
    width = 10,
    height = 5
  )
  
  message("Saved map for: ", trait_name)
}

message("All maps complete.")  
