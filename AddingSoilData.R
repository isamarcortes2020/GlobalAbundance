library(geodata)
library(terra)
File <- "C:/Users/cenv1124/Downloads/SoilMaps"
# Download Soil Organic Carbon (5-15cm) for a region




#soil exchange capacity
Cec5 <- soil_world(var="cec", depth=5, stat="mean",path = File)
Cec15 <- soil_world(var="cec", depth=15, stat="mean",path = File)
Cec30 <- soil_world(var="cec", depth=30, stat="mean",path = File)
# stack them
s <- c(Cec5, Cec15, Cec30)
cec_avg <- mean(s, na.rm = TRUE)


###clay percentage
Clay5 <- soil_world(var="clay", depth=5, stat="mean",path = File)
Clay15 <- soil_world(var="clay", depth=15, stat="mean",path = File)
Clay30 <- soil_world(var="clay", depth=30, stat="mean",path = File)
s <- c(Clay5, Clay15, Clay30)
clay_avg <- mean(s, na.rm = TRUE)



###sand percentage
Sand5 <- soil_world(var="sand", depth=5, stat="mean",path = File)
Sand15 <- soil_world(var="sand", depth=15, stat="mean",path = File)
Sand30 <- soil_world(var="sand", depth=30, stat="mean",path = File)
s <- c(Sand5, Sand15, Sand30)
sand_avg <- mean(s, na.rm = TRUE)

###pH 
pH5 <- soil_world(var="phh2o", depth=5, stat="mean",path = File)
pH15 <- soil_world(var="phh2o", depth=15, stat="mean",path = File)
pH30 <- soil_world(var="phh2o", depth=30, stat="mean",path = File)
s <- c(pH5, pH15, pH30)
pH_avg <- mean(s, na.rm = TRUE)


AddData <- function(file, cec, clay, sand, pH){
  points <- vect(file, geom = c("Coords_x", "Coords_y"), crs = "EPSG:4326")
  points <- project(points, crs(cec))
  
  ####extracting values
  cec_values <- extract(cec, points)[,2]
  clay_values <- extract(clay, points)[,2]
  sand_values <- extract(sand, points)[,2]
  pH_values <- extract(pH, points)[,2]
  
  
  ####adding to dataframe
  file$CEC <- cec_values
  file$Clay <-clay_values
  file$Sand <- sand_values
  file$pH <- pH_values
  return(file)
}



Amax <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/Amax.csv",check.names = FALSE)
Amax <- AddData(Amax,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(Amax, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/Amax.csv", row.names = FALSE)

Asat <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/Asat.csv",check.names = FALSE)
Asat <- AddData(Asat,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(Asat, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/Asat.csv", row.names = FALSE)

BranchHydraulic <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/BranchHydraulic.csv",check.names = FALSE)
BranchHydraulic <- AddData(BranchHydraulic,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(BranchHydraulic, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/BranchHydraulic.csv", row.names = FALSE)

FruitLength <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/FruitLength.csv",check.names = FALSE)
FruitLength <- AddData(FruitLength,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(FruitLength, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/FruitLength.csv", row.names = FALSE)

LeafCaContent <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/LeafCaContent.csv",check.names = FALSE)
LeafCaContent <- AddData(LeafCaContent,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafCaContent, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/LeafCaContent.csv", row.names = FALSE)

LeafCContent <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/LeafCContent.csv",check.names = FALSE)
LeafCContent <- AddData(LeafCContent,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafCContent, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/LeafCContent.csv", row.names = FALSE)

LeafCaContent <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/LeafCaContent.csv",check.names = FALSE)
LeafCaContent <- AddData(LeafCaContent,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafCaContent, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/LeafCaContent.csv", row.names = FALSE)

LeafFreshMass <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/LeafFreshMass.csv",check.names = FALSE)
LeafFreshMass <- AddData(LeafFreshMass,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafFreshMass, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/LeafFreshMass.csv", row.names = FALSE)

LeafMgContent <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/LeafMgContent.csv",check.names = FALSE)
LeafMgContent <- AddData(LeafMgContent,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafMgContent, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/LeafMgContent.csv", row.names = FALSE)

StemDryMass <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/StemDryMass.csv",check.names = FALSE)
StemDryMass <- AddData(StemDryMass,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(StemDryMass, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/StemDryMass.csv", row.names = FALSE)

VesselLumen <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/VesselLumen.csv",check.names = FALSE)
VesselLumen <- AddData(VesselLumen,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(VesselLumen, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/VesselLumen.csv", row.names = FALSE)

LeafHydraulic <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/LeafHydraulic.csv",check.names = FALSE)
LeafHydraulic <- AddData(LeafHydraulic,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafHydraulic, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/LeafHydraulic.csv", row.names = FALSE)
