library(tidyverse)
library(readxl)
library(sf)
library(zoo)
library(arrow)

setwd("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis")

###
# Read data
###

mortality <- read_parquet("data/mortality/mortality_panel.parquet") %>%
    rename(CC_2r = mun_id) %>% mutate(CC_2r = as.character(CC_2r))

hospitalizations <- read_parquet("data/health/hospitalizations.parquet") %>%
    mutate(CC_2r = as.character(CC_2r))

population <- read_csv("data/misc/raw/population.csv") %>%
    reframe(year = ano, CC_2r = id_municipio %>% as.character() %>% str_sub(1,6), population = populacao)

births <- read_csv("data/mortality/raw/births.csv") %>%
    reframe(year = ano, CC_2r = id_municipio_nascimento %>% paste() %>% str_sub(1, 6), total_births)

deforestation <- read_parquet("data/land_cover/deforestation_municipalities.parquet")
climate <- read_parquet("data/climate/climate_data.parquet") %>%
    mutate(CC_2r = as.character(CC_2r)) %>% select(-`__index_level_0__`)

cloud_cover_DETER <- read_parquet("data/cloud_cover/cloud_cover_DETER.parquet") %>% 
    mutate(CC_2r = as.character(CC_2r)) %>% rename("cloud_cover_DETER" = "cloud_cover")

control_variables <- read_parquet("data/misc/control_variables.parquet") %>% mutate(CC_2r = as.character(CC_2r))

municipalities <- st_read("data/boundaries/gadm41_BRA_2.json") %>%
    mutate(CC_2r = CC_2 %>% as.character() %>% str_sub(1, 6))
    
municipalities_simplified <- municipalities %>% st_make_valid() %>% st_simplify(dTolerance = 0.01)

legal_amazon <- read_excel("data/boundaries/Municipios_da_Amazonia_Legal_2022.xlsx") %>%
    reframe(CC_2r = str_sub(CD_MUN, 1, 6)) %>% pull()

immediate_regions <- read_excel("data/boundaries/regioes_geograficas_composicao_por_municipios_2017_20180911.xlsx") %>%
    reframe(CC_2r = str_sub(CD_GEOCODI, 1, 6), CC_i = str_sub(cod_rgi, 1, 6))

###
# Preprocessing
###

# general mortality
mortality_yy <- mortality %>% 
    filter(age_group %in% c("total")) %>%
    group_by(CC_2r, year) %>%
    summarise(deaths = sum(deaths, na.rm = TRUE)) %>%
    ungroup() %>%
    full_join(population, by = c("CC_2r", "year")) %>%
    group_by(CC_2r) %>%
    mutate(mortality_rate_tot = deaths / population,
           mortality_rate_tot_5y = rollsum(deaths, k = 5, fill = NA, align = "right") / population) %>%
    ungroup()

# child mortality
mortality_l5 <- mortality %>% 
    filter(age_group %in% c("under_1", "1_to_4")) %>%
    group_by(CC_2r, year) %>%
    summarise(deaths = sum(deaths, na.rm = TRUE)) %>%
    ungroup() %>%
    full_join(population, by = c("CC_2r", "year")) %>%
    group_by(CC_2r) %>%
    mutate(mortality_rate_l5 = deaths / population,
           mortality_rate_l5_5y = rollsum(deaths, k = 5, fill = NA, align = "right") / population) %>%
    ungroup()

# infant mortality
mortality_l1 <- mortality %>% 
    filter(age_group %in% c("under_1")) %>%
    group_by(CC_2r, year) %>%
    summarise(deaths = sum(deaths, na.rm = TRUE)) %>%
    ungroup() %>%
    full_join(births, by = c("CC_2r", "year")) %>%
    mutate(total_births = replace_na(total_births, 0)) %>%
    group_by(CC_2r) %>%
    mutate(mortality_rate_l1 = deaths / total_births * 1000,
           mortality_rate_l1_5y = rollsum(deaths, k = 5, fill = NA, align = "right") / rollsum(total_births, k = 5, fill = NA, align = "right")) %>%
    ungroup()

# combine mortality data
mortality_full <- mortality_yy %>%
    full_join(mortality_l5, by = c("CC_2r", "year")) %>%
    full_join(mortality_l1, by = c("CC_2r", "year")) %>%
    select(CC_2r, year, mortality_rate_tot, mortality_rate_tot_5y, mortality_rate_l5, mortality_rate_l5_5y, mortality_rate_l1, mortality_rate_l1_5y)

# hospitalization and expenditure rates
hospitalizations <- hospitalizations %>%
    left_join(population, by = c("CC_2r", "year")) %>%
    reframe(CC_2r, year, hosp_rate = hospitalizations / population, ex_pop = total_value / population)

# merge control variables with population information
control_variables <- control_variables %>% left_join(population, by = c("CC_2r", "year"))

# filter for analysis
control_variables <- control_variables %>%
    arrange(CC_2r, year) %>%
    mutate(CC_2r = as.character(CC_2r),
           gdp_pc = gdp / population,
           urban_share = urban_population / population,
           clean_water_share = urban_population_served_water / urban_population) %>%
    select(CC_2r, year, gdp_pc, clean_water_share, educ_ideb, vaccination_index_5y, health_primary_care_coverage, health_doctors_1000, urban_share, clean_water_share)

###
# Compile analysis data
###

# an implementation of cummean that ignores NA values
cummean_na <- function(x) {
  # Replace NA with 0 for cumulative sums and count non-NA values
  not_na <- !is.na(x)
  cumsum_x <- cumsum(ifelse(not_na, x, 0))
  cumcount_x <- cumsum(not_na)
  
  # Calculate cumulative mean ignoring NAs
  cummean_x <- cumsum_x / cumcount_x
  
  # Return the result
  return(cummean_x)
}

# compile analysis data
analysis <- mortality_full %>%
    # combine
    full_join(., deforestation, by = c("CC_2r", "year")) %>%
    left_join(., climate, by = c("CC_2r", "year")) %>%
    left_join(., cloud_cover_DETER, by = c("CC_2r", "year")) %>%
    left_join(., control_variables, by = c("CC_2r", "year")) %>%
    left_join(., immediate_regions, by = "CC_2r") %>%
    left_join(., hospitalizations, by = c("CC_2r", "year")) %>%
    arrange(CC_2r, year) %>% drop_na(CC_2r) %>%
    mutate(region_year = paste(CC_i, year, sep = "_")) %>%
    group_by(CC_2r) %>%
    mutate(
        across(deforestation:deforestation_m, ~ . %>% replace_na(0) %>% cumsum(), .names = "{.col}_cum"),
        across(c(cloud_cover, cloud_cover_DETER), ~ . %>% cummean_na(), .names = "{.col}_cum")
    ) %>%
    ungroup()

# subset legal Amazon and DETER years
analysis_subset <- analysis %>%
    filter(year %in% 2005:2020, CC_2r %in% legal_amazon) %>%
    group_by(CC_2r) %>%
    mutate(across(deforestation_cum:deforestation_m_cum, ~ . - .[year == 2005], .names = "{.col}_r2005")) %>%
    ungroup()

###
# Aggregate deforestation upstream
###

# import spatial weights
weights <- Matrix::readMM(paste0("data/river_network/weights_matrix_", "exponential", "_", "1000", ".mtx"))
weights_municipalities <- read_csv("data/river_network/weights_municipalities.csv")
rownames(weights) <- weights_municipalities$cc_2r; colnames(weights) <- weights_municipalities$cc_2r

# Function to extract weights for a given CC_2r
extraction_worker <- function(CC_2r){tmp <- weights[CC_2r,]; tmp[tmp > 0]}

# Function to calculate weighted sum for all variables
weighted_sum_worker <- function(variable, upstream_weights, CC_2r) {
    map_dbl(upstream_weights, ~ sum(variable[CC_2r %in% names(.x)] * .x[names(.x) %in% CC_2r], na.rm=TRUE))
}

analysis_subset <- analysis_subset %>%
    group_by(year) %>%
    mutate(
        upstream_weights = map(CC_2r, extraction_worker),
        cloud_cover_cum_upstream = weighted_sum_worker(cloud_cover_cum, upstream_weights, CC_2r),
        cloud_cover_DETER_cum_upstream = weighted_sum_worker(cloud_cover_DETER_cum, upstream_weights, CC_2r),
        deforestation_cum_r2005_upstream = weighted_sum_worker(deforestation_cum_r2005, upstream_weights, CC_2r),
        deforestation_p_cum_r2005_upstream = weighted_sum_worker(deforestation_p_cum_r2005, upstream_weights, CC_2r),
        deforestation_a_cum_r2005_upstream = weighted_sum_worker(deforestation_a_cum_r2005, upstream_weights, CC_2r),
        deforestation_u_cum_r2005_upstream = weighted_sum_worker(deforestation_u_cum_r2005, upstream_weights, CC_2r),
        deforestation_m_cum_r2005_upstream = weighted_sum_worker(deforestation_m_cum_r2005, upstream_weights, CC_2r),
        total_upstream = weighted_sum_worker(total, upstream_weights, CC_2r)
    ) %>%
    ungroup() %>%
    select(-upstream_weights)

analysis_subset <- analysis_subset %>%
    # cumulation
    group_by(CC_2r) %>%
    mutate(
        across(forest:mining, ~ . / total, .names = "{.col}_rate"),
        across(deforestation_cum_r2005_upstream:deforestation_m_cum_r2005_upstream, ~ . / total_upstream, .names = "{.col}_rate")
        ) %>%
    ungroup()

###
# Helper functions
###

check_balance <- function(data, group_var, time_var) {
  # Check the balance of the resulting data
  balanced_check <- data %>%
    group_by({{group_var}}, {{time_var}}) %>%
    summarise(count = n(), .groups = 'drop') %>%
    pivot_wider(names_from = {{time_var}}, values_from = count)
  
  # If any NA values exist, it means the panel is not balanced
  if(any(is.na(balanced_check))) {
    return(FALSE)
  } else {
    return(TRUE)
  }
}

get_full_panel_muns <- function(data, variable_set){
    # count unique temporal observations for each municipality in all variables of the variable set
    tmp <- data %>%
        group_by(CC_2r) %>%
        summarise(across(all_of(variable_set), ~ (is.na(.) | is.infinite(.)) %>% `!` %>% sum()))

    # get the median number of observations
    threshold <- tmp %>% summarise(across(where(is.numeric), median)) %>% t() %>% min()

    # filter for municipalities with the median number of observations
    tmp %>%
        filter(if_all(all_of(variable_set), ~ . >= threshold)) %>%
        pull(CC_2r)
}


###
# Compile balanced panels
###

## Compile small panel (2010-2020)

variable_names <- c("mortality_rate_tot", "hosp_rate", "deforestation_cum_r2005_upstream_rate", "cloud_cover_DETER_cum_upstream", "temperature", "precipitation", "gdp_pc", "educ_ideb", "vaccination_index_5y", "health_primary_care_coverage", "health_doctors_1000")
small_panel <- analysis_subset %>%
    filter(year %in% 2010:2020) %>%
    filter(CC_2r %in% get_full_panel_muns(., variable_names)) %>%
    rename(municipality = CC_2r)

small_panel %>% check_balance(municipality, year)

small_panel %>% write_csv("data/analysis/small_panel.csv")

## Compile small panel for infant mortality (2010-2020)

variable_names <- c("mortality_rate_l1", "hosp_rate", "deforestation_cum_r2005_upstream_rate", "cloud_cover_DETER_cum_upstream", "temperature", "precipitation", "gdp_pc", "educ_ideb", "vaccination_index_5y", "health_primary_care_coverage", "health_doctors_1000")
small_panel_l1 <- analysis_subset %>%
    filter(year %in% 2010:2020) %>%
    filter(CC_2r %in% get_full_panel_muns(., variable_names)) %>%
    rename(municipality = CC_2r)

small_panel_l1 %>% check_balance(municipality, year)

small_panel_l1 %>% write_csv("data/analysis/small_panel_l1.csv")

## Compile large panel (2005-2020)

variable_names <- c("mortality_rate_tot", "deforestation_cum_r2005_upstream_rate", "cloud_cover_DETER_cum_upstream", "temperature", "precipitation", "gdp_pc", "educ_ideb", "vaccination_index_5y")
large_panel <- analysis_subset %>%
    filter(year %in% 2005:2020) %>%
    filter(CC_2r %in% get_full_panel_muns(., variable_names)) %>%
    rename(municipality = CC_2r)

large_panel %>% check_balance(municipality, year)

large_panel %>% write_csv("data/analysis/large_panel.csv")

## Compile large panel for infant mortality (2005-2020)

variable_names <- c("mortality_rate_l1", "deforestation_cum_r2005_upstream_rate", "cloud_cover_DETER_cum_upstream", "temperature", "precipitation", "gdp_pc", "educ_ideb", "vaccination_index_5y")
large_panel_l1 <- analysis_subset %>%
    filter(year %in% 2005:2020) %>%
    filter(CC_2r %in% get_full_panel_muns(., variable_names)) %>%
    rename(municipality = CC_2r)

large_panel_l1 %>% check_balance(municipality, year)

large_panel_l1 %>% write_csv("data/analysis/large_panel_l1.csv")


## Plot variable temporal ranges

all_variables <- c("mortality_rate_tot", "mortality_rate_l1", "hosp_rate", "ex_pop", "deforestation", "cloud_cover", "cloud_cover_DETER", "temperature", "precipitation", "gdp_pc", "educ_ideb", "vaccination_index_5y", "health_primary_care_coverage", "health_doctors_1000")
all_variables_labels <- c("Mortality rate (total)", "Mortality rate (under 1)", "Hospitalization rate", "Expenditure per capita", "Deforestation rate", "Cloud cover", "Cloud cover (DETER)", "Temperature", "Precipitation", "GDP per capita", "Education (IDEB)", "Vaccination index (5y)", "Primary care coverage", "Doctors per 1000")

analysis %>%
    summarise(across(c(all_of(all_variables)), ~ list(range(year[!is.na(.)] %>% as.numeric())))) %>%
    pivot_longer(cols = everything(), names_to = "variable", values_to = "range") %>%
    mutate(ymin = map_dbl(range, ~ min(.)),
           ymax = map_dbl(range, ~ max(.))) %>%
    ggplot(aes(x = ymin, xend = ymax, y = factor(variable), yend = factor(variable))) +
    geom_segment(linewidth = 5, lineend = 'round') +
    scale_y_discrete(limits = rev(all_variables), labels = rev(all_variables_labels)) +
    labs(x = "Year", y = NULL) +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 12), axis.text.x = element_text(size = 12), axis.title = element_text(size = 14))

ggsave("output/figures/variable_temporal_ranges.png", width = 3, height = 5, dpi = 300)

# Plot panels on map

# compute combined boundary of legal Amazon
legal_amazon_boundary <- municipalities_simplified %>% filter(CC_2r %in% legal_amazon) %>% st_union() %>% st_boundary()
# get individual line strings
legal_amazon_boundary <- st_cast(legal_amazon_boundary, "MULTILINESTRING") %>% st_cast("LINESTRING")
# get longest line string
legal_amazon_boundary <- legal_amazon_boundary[st_length(legal_amazon_boundary) == max(st_length(legal_amazon_boundary))]

large_panel_plot <- ggplot() +
    geom_sf(data = municipalities_simplified, fill = "grey90", color = "black") +
    geom_sf(data = municipalities_simplified %>% filter(CC_2r %in% large_panel$municipality), aes(fill = "In Sample"), color = "black") +
    geom_sf(data = legal_amazon_boundary, fill = NA, aes(color = "Legal Amazon"), linewidth = 2) +
    scale_color_manual(values = c("Legal Amazon" = "purple"), name = "") +
    scale_fill_manual(values = c("In Sample" = "yellow"), name = "") +
    theme_minimal() +
    theme(legend.position = "inside", legend.position.inside = c(.9, .3))
    
ggsave("output/figures/large_panel_map.png", plot = large_panel_plot, width = 7, height = 7, dpi = 300)

small_panel_plot <- ggplot() +
    geom_sf(data = municipalities_simplified, fill = "grey90", color = "black") +
    geom_sf(data = municipalities_simplified %>% filter(CC_2r %in% small_panel$municipality), aes(fill = "In Sample"), color = "black") +
    geom_sf(data = legal_amazon_boundary, fill = NA, aes(color = "Legal Amazon"), linewidth = 2) +
    scale_color_manual(values = c("Legal Amazon" = "purple"), name = "") +
    scale_fill_manual(values = c("In Sample" = "yellow"), name = "") +
    theme_minimal() +
    theme(legend.position = "inside", legend.position.inside = c(.9, .3))

ggsave("output/figures/small_panel_map.png", plot = small_panel_plot, width = 7, height = 7, dpi = 300)


###
# Exploratory Scatter Plots
###

ggplot(large_panel, aes(x = deforestation_cum_r2005_upstream_rate, y = mortality_rate_tot)) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE) +
    facet_wrap(~year) +
    theme_minimal()
