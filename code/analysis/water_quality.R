library(tidyverse)
library(arrow)
library(lfe)


# read data
water_quality <- read_parquet("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/water_quality/quality_indicators_panel.parquet")

land_cover <- read_parquet("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/land_cover/land_cover_stations.parquet")

# merge data
analysis <- left_join(water_quality, land_cover, by = c("station", "year"))

###
# Helper Functions
###

extract_results <- function(x, outcome=NULL) {
    tmp <- x$model %>% map_df(~ .x %>% broom::tidy())
    if(is.null(outcome)){
        return(tmp %>% select(coefficient = estimate, s_d = std.error, p_value = p.value))
    } else {
        return(tmp %>% filter(term == outcome) %>% select(coefficient = estimate, s_d = std.error, p_value = p.value))
    }
}

analysis_results$model %>% pluck(1) %>% broom::tidy()

plot_results <- function(x, limits = NULL, multiple_ind_var = NULL) {
    if (is.null(limits)) {
        limits <- levels(x$bins)
    }
    if (!is.null(multiple_ind_var)) {
        ggplot(aes(x = bins, y = coefficient), data = x) +
            geom_point() +
            geom_errorbar(aes(ymin = coefficient - 1.96 * s_d, ymax = coefficient + 1.96 * s_d)) +
            geom_hline(yintercept = 0, linetype = "dashed") +
            facet_grid(reformulate(multiple_ind_var, "dep_var"), scales = "free") +
            scale_x_discrete(
                labels = paste0(seq(20, 200, 20), "km"),
                limits = limits
            ) +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
            labs(x = "Upstream Query Distance", y = "Coefficient")
    } else {
    return(
        ggplot(aes(x = bins, y = coefficient), data = x) +
            geom_point() +
            geom_errorbar(aes(ymin = coefficient - 1.96 * s_d, ymax = coefficient + 1.96 * s_d)) +
            geom_hline(yintercept = 0, linetype = "dashed") +
            facet_grid(rows = vars(dep_var), scales = "free_y") +
            scale_x_discrete(
                labels = paste0(seq(20, 200, 20), "km"),
                limits = limits
            ) +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
            labs(x = "Upstream Query Distance", y = "Coefficient")
    )
    }
}

!!sym("analysis")

###
# Cumulative Analysis
###

# compute cumulative sums
analysis_cumulative <- analysis %>%
    arrange(station, year, bins) %>%
    group_by(station, year) %>%
    mutate(across(deforestation:total, cumsum)) %>%
    ungroup() %>%
    group_by(station, bins) %>%
    mutate(across(deforestation:deforestation_m, cumsum)) %>%
    ungroup()

# calculate additional variables
analysis_cumulative <- analysis_cumulative %>% mutate(forest_share = forest / total, deforestation_share = deforestation / total)

# analysis_cumulative = analysis_cumulative %>%
#     mutate(bins = ordered(bins, levels = levels(analysis_cumulative$bins)))
levels(analysis_cumulative$bins) <- c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)")

# prepare analysis grid
analysis_grid <- expand_grid(
    dep_var = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates"),
    bins = levels(analysis_cumulative$bins)
)

## Deforestation; No FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "deforestation")

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_deforestation_Nfe_Ncl.png", width = 5, height = 10, bg = "white")

## Deforestation; One-Way FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation | station")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "deforestation")

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_deforestation_Ofe_Ncl.png", width = 5, height = 10, bg = "white")

## Deforestation; One-Way FE; Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation | station | 0 | station")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "deforestation")

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_deforestation_Ofe_Ycl.png", width = 5, height = 10, bg = "white")

### TODO:
# show limited variance after two-way clustering
# restrict to rural sample; no urbanizations upstream

## Deforestation; FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation | station + year")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "deforestation")

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_deforestation_Yfe_Ncl.png", width = 5, height = 10, bg = "white")

## Deforestation; FE; Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation | station + year | 0 | station + year")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "deforestation")

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_deforestation_Yfe_Ycl.png", width = 5, height = 10, bg = "white")


## Deforestation - by replacement; FE; No Cluster

analysis_grid_replacement <- expand_grid(
    dep_var = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates"),
    ind_var = c("deforestation", "deforestation_f", "deforestation_p", "deforestation_u", "deforestation_m"),
    bins = levels(analysis_cumulative$bins),
)

analysis_results <- analysis_grid_replacement %>%
    mutate(model = pmap(list(dep_var, ind_var, bins), function(.dep_var, .ind_var, .bins) felm(as.formula(paste0(.dep_var, " ~ ", .ind_var, " | station + year")), data = analysis_cumulative %>% filter(bins == .bins))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results)

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"), multiple_ind_var = "ind_var")

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_deforestation_replacement_Nfe_Ncl.png", width = 10, height = 20, bg = "white")

ggplot(aes(x = bins, y = coefficient), data = analysis_results) +
    geom_point() +
    geom_errorbar(aes(ymin = coefficient - 1.96 * s_d, ymax = coefficient + 1.96 * s_d)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    ggh4x::facet_grid2(vars(dep_var), vars(ind_var), scales = "free_y", independent = "y") +
    scale_x_discrete(
        labels = paste0(seq(20, 200, 20), "km"),
        limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"),
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    labs(x = "Upstream Query Distance", y = "Coefficient")


## Forest Share; No FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ forest_share")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "forest_share")

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_forest_share_Nfe_Ncl.png", width = 5, height = 10, bg = "white")

## Forest Share; FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ forest_share | station + year")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "forest_share")

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_forest_share_Yfe_Ncl.png", width = 5, height = 10, bg = "white")



felm(pH ~ forest | station + year | 0 | station + year, data = analysis %>% filter(bins == "[0.0,20000.0)")) %>% broom::tidy(.)
