library('BIEN')
library('dplyr')
library('rtry')
library('readr')
library('readxl')
library('stringr')



fileBIEN <- BIEN_list_all()###all species in BIEN dataset

GlobalData <- read.csv('C:/Users/cenv1124/Downloads/MasterList.csv')
names(GlobalData)[1] <- 'AccSpeciesName'

names(fileBIEN)[1]<-'AccSpeciesName'###renaming one dataset to join

result <- inner_join(fileBIEN, GlobalData, by = "AccSpeciesName")####Here's the join only keeping the overlapping species

trait_list <- BIEN_trait_list()###gets traits in BIEN data
c <- unlist(trait_list)

species_vector <-c(result$AccSpeciesName)
traitTableBIEN <- BIEN_trait_traitbyspecies(trait=c,species=species_vector)

fileTRY <- rtry_import('C:/Users/cenv1124/Downloads/42228_17062025043509/42228.txt')
result_TRY <- semi_join(fileTRY, GlobalData, by = "AccSpeciesName")####Here's the join only keeping the overlapping species

TryDataCleaned <- result_TRY%>%select('AccSpeciesName','TraitName','OrigValueStr','OrigUnitStr')
BienDataCleaned <- traitTableBIEN%>%select('scrubbed_species_binomial','trait_name','trait_value','unit')


names(BienDataCleaned)[1] <- 'SpeciesName'
names(BienDataCleaned)[2]<- 'trait_name'
names(BienDataCleaned)[3]<- 'trait_value'
names(BienDataCleaned)[4]<- 'unit'
BienDataCleaned$DatasetName <- 'BIEN'
BienDataCleaned$trait <- paste(BienDataCleaned$trait_name,BienDataCleaned$unit)
BienDataCleaned <- BienDataCleaned[, -c(2,4)]



names(TryDataCleaned)[1] <- 'SpeciesName'
names(TryDataCleaned)[2]<- 'trait_name'
names(TryDataCleaned)[3]<- 'trait_value'
names(TryDataCleaned)[4]<- 'unit'
TryDataCleaned$DatasetName <- 'TRY'
TryDataCleaned <- TryDataCleaned[TryDataCleaned$trait_name != "" & !is.na(TryDataCleaned$trait_name), ]
TryDataCleaned <- TryDataCleaned[TryDataCleaned$trait_value != "" & !is.na(TryDataCleaned$trait_value), ]
TryDataCleaned$trait <- paste(TryDataCleaned$trait_name,TryDataCleaned$unit)
TryDataCleaned <- TryDataCleaned[, -c(2,4)]



AdditionalData <- read_excel('C:/Users/cenv1124/Downloads/2_TraitsNumeric_SppLevel_240323.xlsx')
AdditionalDataCleaned <- AdditionalData%>%select('Species','Trait','Value')
AdditionalDataCleaned$DatasetName <- 'InternalData'
names(AdditionalDataCleaned)[1] <- 'SpeciesName'
names(AdditionalDataCleaned)[3]<- 'trait_value'
names(AdditionalDataCleaned)[2] <- 'trait'

names(GlobalData)[1] <-'SpeciesName'

mergedData <- rbind(TryDataCleaned,BienDataCleaned,AdditionalDataCleaned)
# Keep only matching species
FinalData <- mergedData %>%
  semi_join(GlobalData, by = "SpeciesName")


FinalData <- FinalData %>%
  filter(trait != "whole plant growth form NA")
FinalData <- FinalData %>%
  filter(trait != "yes")
FinalData <- FinalData %>%
  filter(trait != "no")



FinalData <- unique(FinalData)

df_clean <- FinalData[!grepl("[a-zA-Z]", FinalData$trait_value), ]
df_clean$trait_value <- iconv(df_clean$trait_value, from = "latin1", to = "UTF-8")

df_clean$trait_value <- as.numeric(df_clean$trait_value)
df_clean <- na.omit(df_clean)




standardize_traits <- function(df) {
  
  df %>%
    rowwise() %>%
    mutate(
      
      # --------------------------------------------------
      # 1. Assign each messy trait name to a clean category
      # --------------------------------------------------
      trait_clean = case_when(
        
        # Leaf thickness
        str_detect(trait, regex("leaf thickness", ignore_case = TRUE)) ~ "Leaf thickness (mm)",
        
        str_detect(trait, regex("thickness_mm", ignore_case = TRUE)) ~ "Leaf thickness (mm)",
        # Leaf nitrogen per dry mass
        str_detect(trait, regex("nitrogen", ignore_case = TRUE)) ~ "Leaf N content (mg/g)",
        
        # Leaf phosphorus per dry mass
        str_detect(trait, regex("phosphorus|p content", ignore_case = TRUE)) ~ "Leaf P content (mg/g)",
        
        # SLA / leaf area per dry mass
        str_detect(trait, regex("sla|per leaf dry mass", ignore_case = TRUE)) ~ "SLA (cm2/g)",
        
        # Leaf area (individual)
        str_detect(trait, regex("leaf area(?! per plant)", ignore_case = TRUE)) ~ "Leaf area (cm2)",
        
        # Leaf area per plant
        str_detect(trait, regex("leaf area per plant", ignore_case = TRUE)) ~ "Leaf area per plant (m2)",
        
        # Carbon, Ca, Mg, K, N
        str_detect(trait, regex("carbon", ignore_case = TRUE)) ~ "Leaf C content (%)",
        str_detect(trait, regex("ca_leaf_percent", ignore_case = TRUE)) ~ "Leaf Ca content (%)",
        str_detect(trait, regex("mg_leaf", ignore_case = TRUE)) ~ "Leaf Mg content (%)",
        str_detect(trait, regex("k_leaf", ignore_case = TRUE)) ~ "Leaf K content (%)",
        
        # Leaf masses
        str_detect(trait, regex("leaf dry mass", ignore_case = TRUE)) ~ "Leaf dry mass (g)",
        str_detect(trait, regex("leaf fresh mass", ignore_case = TRUE)) ~ "Leaf fresh mass (g)",
        
        # Plant height
        str_detect(trait, regex("whole plant height", ignore_case = TRUE)) ~ "Plant height (m)",
        str_detect(trait, regex("minimum whole plant height", ignore_case = TRUE)) ~ "Minimum height (m)",
        str_detect(trait, regex("maximum whole plant height", ignore_case = TRUE)) ~ "Maximum height (m)",
        
        # Stem & root masses
        str_detect(trait, regex("stem dry mass", ignore_case = TRUE)) ~ "Stem dry mass (kg)",
        str_detect(trait, regex("root dry mass", ignore_case = TRUE)) ~ "Root dry mass (g)",
        
        # Widths & lengths
        str_detect(trait, regex("minimum leaf width", ignore_case = TRUE)) ~ "Leaf width min (cm)",
        str_detect(trait, regex("maximum leaf width", ignore_case = TRUE)) ~ "Leaf width max (cm)",
        str_detect(trait, regex("minimum leaf length", ignore_case = TRUE)) ~ "Leaf length min (cm)",
        str_detect(trait, regex("maximum leaf length", ignore_case = TRUE)) ~ "Leaf length max (cm)",
        str_detect(trait, regex("fruit length", ignore_case = TRUE)) ~ "Fruit length (mm)",
        str_detect(trait, regex("seed length", ignore_case = TRUE)) ~ "Seed length (mm)",
        str_detect(trait, regex("seed mass", ignore_case = TRUE)) ~ "Seed mass (mg)",
        
        # Reproductive traits
        str_detect(trait, regex("flowering duration", ignore_case = TRUE)) ~ "Flowering duration (months)",
        str_detect(trait, regex("fruiting duration", ignore_case = TRUE)) ~ "Fruiting duration (months)",
        str_detect(trait, regex("flowering begin", ignore_case = TRUE)) ~ "Flowering start",
        str_detect(trait, regex("flower pollination", ignore_case = TRUE)) ~ "Pollination syndrome",
        str_detect(trait, regex("fruit type", ignore_case = TRUE)) ~ "Fruit type",
        
        # Vessel traits
        str_detect(trait, regex("vessel lumen", ignore_case = TRUE)) ~ "Vessel lumen area (mm2)",
        str_detect(trait, regex("vessel number", ignore_case = TRUE)) ~ "Vessel number (mm2)",
        
        # Wood density
        str_detect(trait, regex("wood density", ignore_case = TRUE)) ~ "Wood density (g/cm3)",
        
        TRUE ~ trait    # fallback
      ),
      
      # --------------------------------------------------
      # 2. Detect units directly from trait text
      # --------------------------------------------------
      unit = case_when(
        str_detect(trait, "mm") ~ "mm",
        str_detect(trait, "cm") ~ "cm",
        str_detect(trait, "um|µm|microm|micron") ~ "µm",
        str_detect(trait, "mg/g|mg g-1|mg_g-1") ~ "mg/g",
        str_detect(trait, "%") ~ "%",
        str_detect(trait, "g/kg") ~ "g/kg",
        str_detect(trait, "m2|m²") ~ "m2",
        str_detect(trait, "cm2|cm²") ~ "cm2",
        TRUE ~ NA_character_
      ),
      
      # --------------------------------------------------
      # 3. Convert to standard units for each trait group
      # --------------------------------------------------
      trait_value_std = case_when(
        
        # ---- Leaf thickness -> mm ----
        trait_clean == "Leaf thickness (mm)" & unit == "mm" ~ trait_value,
        trait_clean == "Leaf thickness (mm)" & unit == "µm" ~ trait_value / 1000,
        trait_clean == "Leaf thickness (mm)" & unit == "cm" ~ trait_value * 10,
        
        # ---- Leaf N content -> mg/g ----
        trait_clean == "Leaf N content (mg/g)" & unit == "%" ~ trait_value * 10,   # 1% = 10 mg/g
        trait_clean == "Leaf N content (mg/g)" & unit == "g/kg" ~ trait_value / 1,
        trait_clean == "Leaf N content (mg/g)" & unit == "mg/g" ~ trait_value,
        
        # ---- Leaf P content -> mg/g ----
        trait_clean == "Leaf P content (mg/g)" & unit == "%" ~ trait_value * 10,
        trait_clean == "Leaf P content (mg/g)" & unit == "g/kg" ~ trait_value / 1,
        trait_clean == "Leaf P content (mg/g)" & unit == "mg/g" ~ trait_value,
        
        # ---- Leaf area -> cm2 ----
        trait_clean == "Leaf area (cm2)" & unit == "cm2" ~ trait_value,
        trait_clean == "Leaf area (cm2)" & unit == "mm2" ~ trait_value / 100,
        
        # ---- Leaf area per plant -> m2 ----
        trait_clean == "Leaf area per plant (m2)" & unit == "cm2" ~ trait_value / 10000,
        trait_clean == "Leaf area per plant (m2)" & unit == "m2" ~ trait_value,
        
        # (others do not require conversion)
        TRUE ~ trait_value
      )
    ) %>% 
    ungroup()
}





#FinalData$trait_value <- as.numeric(FinalData$trait_value)

cleaned <- standardize_traits(df_clean)

df <- subset(cleaned, select = -trait_value) 
df <- subset(df, select = -trait)
df <- subset(df, select = -unit)

colnames(df) <- c("species","DatasetName","trait","trait_value")

library(dplyr)
library(stringr)

df <- df %>%
  mutate(
    trait_value = if_else(
      str_detect(trait, "\\(mg/g\\)"),
      trait_value / 10,
      trait_value
    ),
    trait = if_else(
      str_detect(trait, "\\(mg/g\\)"),
      str_replace(trait, "\\(mg/g\\)", "(%)"),
      trait
    )
  )

df <- df %>%
  mutate(
    trait = case_when(
      trait %in% c("Stem hydraulic conductivity kg s?1 MPa?1",
                   "Stem hydraulic conductivity kg s-1 MPa-1") ~ "Stem hydraulic conductivity kg s-1 MPa-1",
      TRUE ~ trait
    )
  )

df <- df %>%
  mutate(
    trait = case_when(
      trait %in% c("Branch hydraulic conductance kg/m/s/Mpa",
                   "Branch hydraulic conductance kg.m-1.MPa-1.s-1") ~ "Branch hydraulic conductance kg.m-1.MPa-1.s-1",
      TRUE ~ trait
    )
  )




df <- df %>%
  mutate(
    trait = case_when(
      trait %in% c("Branch hydraulic conductance [kg/(m s Mpa)]",
                   "Branch hydraulic conductance kg.m-1.MPa-1.s-1") ~ "Branch hydraulic conductance kg.m-1.MPa-1.s-1",
      TRUE ~ trait
    )
  )

df <- df %>%
  mutate(
    trait = case_when(
      trait %in% c("Leaf N content (%)",
                   "N_leaf_Percent") ~ "Leaf N content (%)",
      TRUE ~ trait
    )
  )


df <- df %>%
  mutate(
    trait = case_when(
      trait %in% c("Area_cm2",
                   "Leaf area (cm2)") ~ "Leaf area (cm2)",
      TRUE ~ trait
    )
  )


df <- df %>%
  mutate(
    trait = case_when(
      trait %in% c("P_leaf_Percent",
                   "Leaf P content (%)") ~ "Leaf P content (%))",
      TRUE ~ trait
    )
  )

df <- df %>%
  mutate(
    trait = case_when(
      trait %in% c("Leaf fresh mass (g)",
                   "LeafFreshMass_g") ~ "Leaf fresh mass (g)",
      TRUE ~ trait
    )
  )

df <- df %>%
  mutate(
    trait = case_when(
      trait %in% c("Leaf dry mass (g)",
                   "LeafDryMass_g") ~ "Leaf dry mass (g)",
      TRUE ~ trait
    )
  )

df <- df %>%
  filter(trait != "Flowering duration (months)")

df <- df %>%
  filter(trait != "Plant height (m)")

df <- df %>%
  filter(trait != "longest whole plant longevity years")

df <- df %>%
  filter(trait != "maximum whole plant longevity years")

df <- df %>%
  filter(trait != "Leaf length max (cm)")

df <- df %>%
  filter(trait != "Leaf length min (cm)")

df <- df %>%
  filter(trait != "Leaf width min (cm)")

df <- df %>%
  filter(trait != "inflorescence length cm")

df <- df %>%
  filter(trait != "diameter at breast height (1.3 m) cm")

df <- df %>%
  filter(trait != "whole plant primary juvenile period length years")

df <- df %>%
  filter(trait != "Leaf lamina fracture toughness J.m-2")

df <- df %>%
  filter(trait != "Leaf width max (cm)")

df <- df %>%
  mutate(
    trait_value = ifelse(
      trait == "Leaf thickness (mm)" & trait_value > 10,
      trait_value / 1000,
      trait_value
    )
  )


df <- df %>%
  mutate(
    trait_value = ifelse(
      trait == "Leaf P content (%))" & trait_value > 1,
      trait_value / 10,  # likely mg/g → %
      trait_value
    )
  ) %>%
  filter(!(trait == "Leaf P content (%))" & trait_value > 1))


df <- df %>%
  filter(!(trait == "Leaf area (cm2)" & trait_value <= 0))

df <- df %>%
  filter(!(trait == "Seed mass (mg)" & trait_value > 100000))

df <- df %>%
  filter(!(trait == "Leaf N content (%)" & trait_value > 10))



t<- df %>%
  group_by(trait) %>%
  summarise(
    n = n(),
    n_species = n_distinct(species),
    mean = mean(trait_value, na.rm = TRUE),
    sd = sd(trait_value, na.rm = TRUE),
    min = min(trait_value, na.rm = TRUE),
    max = max(trait_value, na.rm = TRUE),
    median = median(trait_value, na.rm = TRUE)
  ) %>%
  arrange(trait)


df <-unique(df)
write.csv(df,"L:/TraitData.csv")
write.csv(t,"L:/TraitDataStats.csv")
