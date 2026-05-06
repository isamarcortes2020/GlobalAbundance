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



Amax <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/Amax2017_Present.csv",check.names = FALSE)
Amax <- AddData(Amax,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(Amax, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/Amax2017_Present.csv", row.names = FALSE)

Asat <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/Asat2017_Present.csv",check.names = FALSE)
Asat <- AddData(Asat,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(Asat, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/Asat2017_Present.csv", row.names = FALSE)


BranchHydraulic2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/BranchHydraulic2017_Present.csv",check.names = FALSE)
BranchHydraulic2017_Present <- AddData(BranchHydraulic2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(BranchHydraulic2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/BranchHydraulic2017_Present.csv", row.names = FALSE)


FruitLength2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/FruitLength2017_Present.csv",check.names = FALSE)
FruitLength2017_Present <- AddData(FruitLength2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(FruitLength2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/FruitLength2017_Present.csv", row.names = FALSE)


LeafCaContent2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafCaContent2017_Present.csv",check.names = FALSE)
LeafCaContent2017_Present <- AddData(LeafCaContent2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafCaContent2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafCaContent2017_Present.csv", row.names = FALSE)


LeafCContent2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/UpdatedDatasetSoFar/LeafCContent.csv",check.names = FALSE)
LeafCContent2017_Present <- AddData(LeafCContent2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafCContent2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafCContent2017_Present.csv", row.names = FALSE)


LeafFreshMass2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafFreshMass2017_Present.csv",check.names = FALSE)
LeafFreshMass2017_Present <- AddData(LeafFreshMass2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafFreshMass2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafFreshMass2017_Present.csv", row.names = FALSE)


LeafHydraulic2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafHydraulic2017_Present.csv",check.names = FALSE)
LeafHydraulic2017_Present <- AddData(LeafHydraulic2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafHydraulic2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafHydraulic2017_Present.csv", row.names = FALSE)


LeafKContent2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafKContent2017_Present.csv",check.names = FALSE)
LeafKContent2017_Present <- AddData(LeafKContent2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafKContent2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafKContent2017_Present.csv", row.names = FALSE)


LeafMgContent2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafMgContent2017_Present.csv",check.names = FALSE)
LeafMgContent2017_Present <- AddData(LeafMgContent2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafMgContent2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafMgContent2017_Present.csv", row.names = FALSE)


LeafPContent2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafPContent2017_Present.csv",check.names = FALSE)
LeafPContent2017_Present <- AddData(LeafPContent2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafPContent2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafPContent2017_Present.csv", row.names = FALSE)


LeafWaterContent2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafWaterContent2017_Present.csv",check.names = FALSE)
LeafWaterContent2017_Present <- AddData(LeafWaterContent2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(LeafWaterContent2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/LeafWaterContent2017_Present.csv", row.names = FALSE)


RootDryMass2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/RootDryMass2017_Present.csv",check.names = FALSE)
RootDryMass2017_Present <- AddData(RootDryMass2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(RootDryMass2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/RootDryMass2017_Present.csv", row.names = FALSE)


SeedLength2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/SeedLength2017_Present.csv",check.names = FALSE)
SeedLength2017_Present <- AddData(SeedLength2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(SeedLength2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/SeedLength2017_Present.csv", row.names = FALSE)


StemDryMass2017_Present <- read.csv("R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/StemDryMass2017_Present.csv",check.names = FALSE)
StemDryMass2017_Present <- AddData(StemDryMass2017_Present,cec_avg,clay_avg,sand_avg,pH_avg)
write.csv(StemDryMass2017_Present, "R:/GlobalDataset/TraitsCombinedWithGeoTessera/2017_Onwards/StemDryMass2017_Present.csv", row.names = FALSE)



