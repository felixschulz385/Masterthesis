library(tidyverse)
library(sf)
library(arrow)
library(sfarrow)
library(lfe)
library(stargazer)

# read data
water_quality <- read_parquet("data/water_quality/quality_indicators_panel.parquet") %>% filter(pH != 0)

land_cover <- read_parquet("data/land_cover/land_cover_stations.parquet")

stations <- st_read_feather("data/water_quality/stations_rivers.feather") %>%
    st_set_crs(5641)

boundaries <- st_read("data/boundaries/gadm41_BRA_0.json") %>% st_transform(5641)

# merge data
analysis <- left_join(water_quality, land_cover, by = c("station", "year"))

###
# Helper Functions
###

dep_vars = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates")

extract_results <- function(x, outcome = NULL) {
    tmp <- x$model %>% map_df(~ .x %>% broom::tidy())
    if (is.null(outcome)) {
        return(tmp %>% select(coefficient = estimate, s_d = std.error, p_value = p.value))
    } else {
        return(tmp %>% filter(term == outcome) %>% select(coefficient = estimate, s_d = std.error, p_value = p.value))
    }
}

plot_results <- function(x, limits = NULL, multiple_ind_var = NULL, facet_labels = NULL) {
    if (is.null(limits)) {
        limits <- levels(x$bins)
    }
    if (is.null(facet_labels)) {
        labeller <- as_labeller(c(
            pH = "pH",
            turbidity = "Turbidity",
            biochem_oxygen_demand = "Biochemical\nOxygen Demand",
            dissolved_oxygen = "Dissolved Oxygen",
            total_residue = "Total Residue",
            total_nitrogen = "Total Nitrogen",
            nitrates = "Nitrates"
        ))
    } else {
        labeller <- as_labeller(facet_labels)
    }
    if (!is.null(multiple_ind_var)) {
        ggplot(aes(x = bins, y = coefficient), data = x) +
            geom_point() +
            geom_errorbar(aes(ymin = coefficient - 1.96 * s_d, ymax = coefficient + 1.96 * s_d)) +
            geom_hline(yintercept = 0, linetype = "dashed") +
            ggh4x::facet_grid2(vars(dep_var), vars(ind_var), scales = "free_y", independent = "y") +
            scale_x_discrete(
                labels = str_extract_all(limits, "\\d+") %>% map_chr(~ paste0(as.double(.x[[1]]) / 1e3, "-", as.double(.x[[3]]) / 1e3, "km")),
                limits = limits
            ) +
            theme_bw() +
            theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
            labs(x = "Upstream Query Distance", y = "Coefficient")
    } else {
        return(
            ggplot(aes(x = bins, y = coefficient), data = x) +
                geom_point() +
                geom_errorbar(aes(ymin = coefficient - 1.96 * s_d, ymax = coefficient + 1.96 * s_d)) +
                geom_hline(yintercept = 0, linetype = "dashed") +
                facet_grid(rows = vars(dep_var), scales = "free_y", labeller = labeller) +
                scale_x_discrete(
                    labels = str_extract_all(limits, "\\d+") %>% map_chr(~ paste0(as.double(.x[[1]]) / 1e3, "-", as.double(.x[[3]]) / 1e3, "km")),
                    limits = limits
                ) +
                theme_bw() +
                theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
                labs(x = "Upstream Query Distance", y = "Coefficient")
        )
    }
}

# function to filter and impute missing values for given variable
filter_impute <- function(x, variable){
    variable_sym <- sym(variable)
    
    x %>%
        group_by(station) %>%
        filter(sum(is.na(!!variable_sym)) / n() <= .05) %>%
        mutate(across(all_of(variable), ~ zoo::na.approx(., na.rm = FALSE))) %>%
        ungroup()
}

###
# General findings
###

## main result

# Calculate the annual-station averages
analysis_single <- analysis %>%
    arrange(station, year, bins) %>%
    group_by(station, bins) %>%
    mutate(across(deforestation:deforestation_m, cumsum)) %>%
    ungroup() %>%
    group_by(station, year) %>%
    summarise(across(pH:nitrates, unique), across(deforestation:total, sum)) %>%
    ungroup() %>%
    filter(total != 0) %>%
    mutate(
        forest_share = forest / total,
        deforestation_share = deforestation / total
    )


# prepare analysis grid
analysis_results <- tibble(
    dep_var = dep_vars
)

analysis_results <- analysis_results %>%
    mutate(formula = map(dep_var, ~ as.formula(paste0(.x, " ~ deforestation_share | station + year | 0 | station + year"))),
           model = map2(formula, dep_var, ~ felm(.x, data = analysis_single %>% filter(year >= 2005) %>% filter_impute(.y))))

# hacky way of forcing stargazer to not obtain the formula from call
analysis_results$model[[1]]$call <- NULL

stargazer(analysis_results$model, 
          type = "latex", 
          covariate.labels = c("\\% Deforestation"), 
          style = "qje",
          header = FALSE,
          out = "output/tables/reg_stations_deforestation_agg_2005.tex")

## sensitivity analysis: early years

analysis_results <- analysis_results %>%
    mutate(model = map2(formula, dep_var, ~ felm(.x, data = analysis_single %>% filter(year < 2005) %>% filter_impute(.y))))

# hacky way of forcing stargazer to not obtain the formula from call
analysis_results$model[[1]]$call <- NULL

stargazer(analysis_results$model, 
          type = "latex",
          covariate.labels = c("% Deforestation"), 
          style = "qje",
          header = FALSE,
          out = "output/tables/reg_stations_deforestation_agg_1970.tex")

# Why is the effect different for early years? What are possible mechanisms?

###
# Effect of distance
###

analysis_single <- analysis %>%
    arrange(station, year, bins) %>%
    group_by(station, bins) %>%
    mutate(across(deforestation:deforestation_m, cumsum)) %>%
    ungroup() %>%
    mutate(bins = forcats::fct_collapse(bins,
                                        "[0.0,100000.0)" = c("[0.0,50000.0)", "[50000.0,100000.0)"),
                                        "[100000.0,200000.0)" = c("[100000.0,150000.0)", "[150000.0,200000.0)"),
                                        "[200000.0,300000.0)" = c("[200000.0,250000.0)", "[250000.0,300000.0)"),
                                        "[300000.0,400000.0)" = c("[300000.0,350000.0)", "[350000.0,400000.0)"),
                                        "[400000.0,500000.0)" = c("[400000.0,450000.0)", "[450000.0,500000.0)"),
                                        "[500000.0,600000.0)" = c("[500000.0,550000.0)", "[550000.0,600000.0)"),
                                        "[600000.0,700000.0)" = c("[600000.0,650000.0)", "[650000.0,700000.0)"),
                                        "[700000.0,800000.0)" = c("[700000.0,750000.0)", "[750000.0,800000.0)"),
                                        "[800000.0,900000.0)" = c("[800000.0,850000.0)", "[850000.0,900000.0)"),
                                        "[900000.0,1000000.0)" = c("[900000.0,950000.0)", "[950000.0,1000000.0)"))) %>%
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
    filter(total != 0, year >= 2005)

# prepare analysis grid
analysis_grid <- expand_grid(
    dep_var = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates"),
    bins = levels(analysis_single$bins)
)

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation_share | station + year | 0 | station + year")), data = analysis_single %>% filter(bins == .y) %>% filter_impute(.x))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results)

analysis_results %>% plot_results(limits = levels(analysis_single$bins))

ggsave("output/figures/stations_deforestation_g5_share_Yfe_Ycl.png", width = 7.5, height = 10, bg = "white")

###
# Effect of alternative land use
###

# Calculate the annual-station averages
analysis_single <- analysis %>%
    arrange(station, year, bins) %>%
    group_by(station, bins) %>%
    mutate(across(deforestation:deforestation_m, cumsum)) %>%
    ungroup() %>%
    group_by(station, year) %>%
    summarise(across(pH:nitrates, unique), across(deforestation:total, sum)) %>%
    ungroup() %>%
    mutate(
        forest_share = forest / total,
        deforestation_share = deforestation / total,
        deforestation_p_share = deforestation_p / total,
        deforestation_a_share = deforestation_a / total,
        deforestation_u_share = deforestation_u / total,
        deforestation_m_share = deforestation_m / total,
    ) %>%
    filter(total != 0)

# prepare analysis grid
analysis_results <- tibble(
    dep_var = dep_vars
)

analysis_results <- analysis_results %>%
    mutate(formula = map(dep_var, ~ as.formula(paste0(.x, " ~ deforestation_p_share + deforestation_a_share + deforestation_u_share + deforestation_m_share | station + year | 0 | station + year"))),
           model = map2(formula, dep_var, ~ felm(.x, data = analysis_single %>% filter(year >= 2005) %>% filter_impute(.y))))

# hacky way of forcing stargazer to not obtain the formula from call
analysis_results$model[[1]]$call <- NULL

stargazer(analysis_results$model, type = "text")

stargazer(analysis_results$model, type = "latex", out = "output/tables/reg_stations_land_use_2005.tex") #, dep.var.labels = analysis_results$dep_var


analysis_results <- analysis_results %>%
    mutate(formula = map(dep_var, ~ as.formula(paste0(.x, " ~ deforestation_p_share + deforestation_a_share + deforestation_u_share + deforestation_m_share | station + year | 0 | station + year"))),
           model = map2(formula, dep_var, ~ felm(.x, data = analysis_single %>% filter(year >= 2005) %>% filter_impute(.y))))

# hacky way of forcing stargazer to not obtain the formula from call
analysis_results$model[[1]]$call <- NULL

stargazer(analysis_results$model, type = "latex", out = "output/tables/reg_stations_land_use_1970.tex") #, dep.var.labels = analysis_results$dep_var
