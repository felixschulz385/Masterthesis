library(tidyverse)
library(sf)

topology <- arrow::read_feather("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/river_network/topology.feather")

topology$estuary %>% sum()
# 1025
topology$confluence %>% sum()
# 123140
topology$source %>% sum()
# 125192

# get the minimum and maximum non-na year for each variable
analysis %>%
    reframe(across(c(deforestation_rate, cloud_cover, cloud_cover_DETER, mortality_rate_tot_5y, hosp_rate, gdp_pc, educ_ideb, vaccination_index_5y, health_doctors_1000, urban_share, clean_water_share), ~ range(year[which(!is.na(.))])))

analysis %>% names()
