library(tidyverse)
library(sf)
library(arrow)
library(sfarrow)
library(kableExtra)
library(lfe)
library(texreg)

# read data
water_quality <- read_parquet("data/water_quality/quality_indicators_panel.parquet") %>% filter(pH != 0)

land_cover <- read_parquet("data/land_cover/land_cover_stations.parquet")

stations <- st_read_feather("data/water_quality/stations_rivers.feather") %>%
    st_set_crs(5641) %>% filter(!duplicated(Codigo))

boundaries <- st_read("data/boundaries/gadm41_BRA_0.json") %>% st_transform(5641)

# merge data
analysis <- left_join(water_quality, land_cover, by = c("station", "year"))


###
# Descriptive statistics
###

# number of stations
water_quality %>% pull(station) %>% unique() %>% length()

# plot of stations
ggplot() +
    geom_sf(data = boundaries) +
    geom_sf(data = stations %>% filter(Codigo %in% unique(water_quality$station)), size = .5) +
    theme_minimal()
ggsave("output/figures/stations.png", width = 5, height = 5, bg = "white")

water_quality %>%
    summarise(across(pH:nitrates, ~ mean(!is.na(.))))

# display histograms of all variables in one plot with stacked axes

# Natural limits
limits <- tibble(
  variable = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates"),
  min_val = c(0, 0, 0, 0, 0, 0, 0),
  max_val = c(14, 100, 50, 200, 500, 100, 10)
)

# Reshape data to long format and merge with limits to get natural limits for each variable
water_quality_long <- water_quality %>%
    pivot_longer(pH:nitrates, names_to = "variable", values_to = "value") %>%
    left_join(limits, by = "variable") %>%
    mutate(variable = factor(variable, levels = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates")))

# Provide summary table for each variable
# Schema: variable, mean, median, min, max, number of missing values (rounded to 2 decimal places)
# format to latex using kableExtra, booktabs, label and caption
water_quality_summary <- water_quality_long %>%
    group_by(variable) %>%
    summarise(min = min(value, na.rm = TRUE),
              mean = mean(value, na.rm = TRUE),
              median = median(value, na.rm = TRUE),
              max = max(value, na.rm = TRUE),
              missing = mean(is.na(value))) %>%
    mutate(across(mean:missing, ~ round(.x, 2))) %>%
    mutate(variable = c("pH", "Turbidity", "Biochemical Oxygen Demand", "Dissolved Oxygen", "Total Residue", "Total Nitrogen", "Nitrates")) %>%
    `colnames<-`(c("", "Min", "Mean", "Median", "Max", "Missing"))


# Generate the LaTeX table string
latex_table <- water_quality_summary %>%
  kbl("latex", booktabs = TRUE, caption = "Summary Statistics of Sensor Data", linesep = "") %>%
  as.character()

# Insert the \label command after the third line
lines <- strsplit(latex_table, "\n")[[1]]
lines <- append(lines, "\\label{tbl-summary-sensor-data}", after = 3)
latex_table_with_label <- paste(lines, collapse = "\n")

# Write the modified LaTeX table string to a file
output_file <- "output/tables/summary_sensor_data.tex"
writeLines(latex_table_with_label, output_file)

###
# Helper Functions
###

dep_vars = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates")
dep_vars_table = c("pH", "Turbidity", "\\makecell{Biochemical\\\\Oxygen\\\\Demand}", "\\makecell{Dissolved\\\\Oxygen}", "\\makecell{Total\\\\Residue}", "\\makecell{Total\\\\Nitrogen}", "Nitrates")

# function to filter and impute missing values for given variable
filter_impute <- function(x, variable){
    variable_sym <- sym(variable)
    
    x %>%
        group_by(station) %>%
        filter(sum(is.na(!!variable_sym)) / n() <= .05) %>%
        mutate(across(all_of(variable), ~ zoo::na.approx(., na.rm = FALSE))) %>%
        ungroup()
}

standardize <- function(x) {
    return((x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE))
}


###
# General findings
###

## main result

# Calculate the annual-station averages
analysis_single <- analysis %>%
    arrange(station, year, bins) %>%
    #filter(bins %in% levels(analysis$bins)[1:5]) %>%
    group_by(station, bins) %>%
    mutate(across(deforestation:deforestation_m, cumsum)) %>%
    ungroup() %>%
    group_by(station, year) %>%
    summarise(across(pH:nitrates, unique), across(deforestation:total, sum)) %>%
    ungroup() %>%
    filter(total != 0) %>%
    mutate(
        forest_share = forest / total,
        deforestation_share = deforestation / total,
        deforestation_p_share = deforestation_p / total,
        deforestation_a_share = deforestation_a / total,
        deforestation_u_share = deforestation_u / total,
        deforestation_m_share = deforestation_m / total,
        forest_share = forest / total,
        pasture_share = pasture / total,
        agriculture_share = agriculture / total,
        urban_share = urban / total,
        mining_share = mining / total,
    ) %>%
    left_join(stations %>% select(station = Codigo, estuary), by = "station") %>%
    mutate(estuary_year = paste0(estuary, "_", year)) %>%
    mutate(across(dep_vars, standardize, .names = "{.col}_sd"), across(deforestation_share:deforestation_m_share, standardize, .names = "{.col}_sd"))


# prepare analysis grid
analysis_results <- tibble(
    dep_var = dep_vars
)


analysis_results <- analysis_results %>%
    mutate(formula = map(dep_var, ~ as.formula(paste0(.x, " ~ I(deforestation_share * 100) | station + year | 0 | estuary + estuary_year"))),
           model = map2(formula, dep_var, ~ felm(.x, data = analysis_single %>% filter(year > 2004) %>% filter_impute(.y))))

texreg(analysis_results$model,
        file = "output/tables/reg_stations_deforestation.tex",
        custom.model.names = dep_vars_table,
        custom.coef.names = c("Deforestation"),
        stars = c(0.01, 0.05, 0.1),
        caption = "Regression Results: Aggregate Deforestation and Pollution",
        label = "tbl-reg-deforestation-sensors",
        include.rsquared = FALSE, include.adjrs = FALSE,
        custom.note = "\\item \\textit{Notes:} Unbalanced panels of all available data post 2004. Models include station and year fixed effects. Standard errors are clustered at the watershed and year level. Significance: %stars",
        booktabs = TRUE,
        use.packages = FALSE,
        threeparttable = TRUE,
        scalebox = 0.8,
        dcolumn = TRUE
)

analysis_results <- analysis_results %>%
    mutate(formula = map(dep_var, ~ as.formula(paste0(.x, "_sd ~ I(deforestation_share * 100) | station + year | 0 | estuary + estuary_year"))),
           model = map2(formula, dep_var, ~ felm(.x, data = analysis_single %>% filter(year > 2004) %>% filter_impute(.y))))

texreg(analysis_results$model,
        file = "output/tables/reg_stations_deforestation_sd.tex",
        custom.model.names = dep_vars_table,
        custom.coef.names = c("Deforestation"),
        stars = c(0.01, 0.05, 0.1),
        caption = "Regression Results: Deforestation and Pollution, Standardized Dependent Variables",
        label = "tbl-reg-deforestation-sensors-sd",
        include.rsquared = FALSE, include.adjrs = FALSE,
        custom.note = "\\item \\textit{Notes:} Unbalanced panels of all available data post 2004. Models include station and year fixed effects. Standard errors are clustered at the watershed and watershed-year level. Significance: %stars",
        booktabs = TRUE,
        use.packages = FALSE,
        threeparttable = TRUE,
        scalebox = 0.8,
        dcolumn = TRUE
)

###
# Effect of replacement
###

analysis_results <- analysis_results %>%
    mutate(formula = map(dep_var, ~ as.formula(paste0(.x, " ~ I(deforestation_p_share * 100) + I(deforestation_a_share * 100) + I(deforestation_u_share * 100) + I(deforestation_m_share * 100) | station + year | 0 | estuary + estuary_year"))),
           model = map2(formula, dep_var, ~ felm(.x, data = analysis_single %>% filter_impute(.y))))

texreg(analysis_results$model,
        file = "output/tables/reg_stations_deforestation_replacement.tex",
        custom.model.names = dep_vars_table,
        custom.coef.names = c("$\\Delta$Pasture", "$\\Delta$Agriculture", "$\\Delta$Urban", "$\\Delta$Mining"),
        stars = c(0.01, 0.05, 0.1),        
        caption = "Regression Results: Deforestation and Pollution, Replacement Effects",
        label = "tbl-reg-deforestation-sensors-replacement",
        include.rsquared = FALSE, include.adjrs = FALSE,
        custom.note = "\\item \\textit{Notes:} Unbalanced panels. Models include station fixed effects. Standard errors are clustered at the watershed and watershed-year level. Significance: %stars",
        booktabs = TRUE,
        use.packages = FALSE,
        threeparttable = TRUE,
        scalebox = 0.7,
        dcolumn = TRUE
)

###
# Levels
###

analysis_results <- analysis_results %>%
    mutate(formula = map(dep_var, ~ as.formula(paste0(.x, " ~ I(forest_share * 100) + I(pasture_share * 100) + I(agriculture_share * 100) + I(urban_share * 100) + I(mining_share * 100) | station + year | 0 | estuary + estuary_year"))),
           model = map2(formula, dep_var, ~ felm(.x, data = analysis_single %>% filter_impute(.y))))

texreg(analysis_results$model,
        file = "output/tables/reg_stations_deforestation_replacement_levels.tex",
        custom.model.names = dep_vars_table,
        custom.coef.names = c("$\\rho$Forest", "$\\rho$Pasture", "$\\rho$Agriculture", "$\\rho$Urban", "$\\rho$Mining"),
        stars = c(0.01, 0.05, 0.1),
        caption = "Regression Results: Deforestation and Pollution, Replacement Effects, Levels",
        label = "tbl-reg-deforestation-sensors-replacement-levels",
        include.rsquared = FALSE, include.adjrs = FALSE,
        custom.note = "\\item \\textit{Notes:} Unbalanced panels. Models include station and year fixed effects. Standard errors are clustered at the watershed and watershed-year level. Significance: %stars",
        booktabs = TRUE,
        use.packages = FALSE,
        threeparttable = TRUE,
        scalebox = 0.7,
        dcolumn = TRUE
)

###
# Effect of distance
###

# aggregate in bins
analysis_bins <- analysis %>%
    arrange(station, year, bins) %>%
    group_by(station, bins) %>%
    mutate(across(deforestation:deforestation_m, cumsum)) %>%
    ungroup() %>%
    group_by(station, year, bins) %>%
    summarise(across(pH:nitrates, unique), across(deforestation:total, sum)) %>%
    ungroup() %>%
    mutate(
        deforestation_share = deforestation / total,
        deforestation_p_share = deforestation_p / total,
        deforestation_a_share = deforestation_a / total,
        deforestation_u_share = deforestation_u / total,
        deforestation_m_share = deforestation_m / total,
    ) %>%
    filter(total != 0) %>%
    left_join(stations %>% select(station = Codigo, estuary), by = "station") %>%
    mutate(estuary_year = paste0(estuary, "_", year)) %>%
    mutate(distance = str_extract(bins, "\\d+") %>% as.double() %>% "/"(1000))

# prepare analysis grid
analysis_results <- tibble(
    dep_var = dep_vars
)

analysis_results <- analysis_results %>%
    mutate(formula = map(dep_var, ~ as.formula(paste0(.x, " ~ deforestation_share + deforestation_share : I(distance / 100) | station + year | 0 | estuary + estuary_year"))),
           model = map2(formula, dep_var, ~ felm(.x, data = analysis_bins %>% filter(year > 2004)  %>% filter_impute(.y))))

texreg(analysis_results$model,
        file = "output/tables/reg_stations_deforestation_distance.tex",
        custom.model.names = dep_vars_table,
        custom.coef.names = c("Deforestation", "Deforestation x Distance"),
        stars = c(0.01, 0.05, 0.1),        
        caption = "Regression Results: Deforestation and Pollution, Effect in Distance",
        label = "tbl-reg-deforestation-sensors-distance",
        include.rsquared = FALSE, include.adjrs = FALSE,
        custom.note = "\\item \\textit{Notes:} Unbalanced panels of all available data post 2004. Models include station fixed effects. Standard errors are clustered at the watershed and watershed-year level. Significance: %stars",
        booktabs = TRUE,
        use.packages = FALSE,
        threeparttable = TRUE,
        scalebox = 0.8,
        dcolumn = TRUE
)
