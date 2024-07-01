library(tidyverse)
library(arrow)
library(lfe)
library(stargazer)

# read data
mortality <- arrow::read_parquet("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/mortality/mortality_panel.parquet")

# child mortality
mortality <- mortality %>% 
    drop_na(deaths, population) %>% 
    filter(age_group %in% c("under_1", "1_to_4")) %>%
    group_by(mun_id, year) %>%
    summarise(mortality_rate = sum(deaths) / sum(population)) %>%
    ungroup()

# general mortality
mortality <- mortality %>% 
    drop_na(deaths, population) %>% 
    group_by(mun_id, year) %>%
    summarise(mortality_rate = sum(deaths) / sum(population)) %>%
    ungroup()

ggplot(mortality %>% filter(mun_id == 120001)) +
    geom_line(aes(x = year, y = mortality_rate))

land_cover <- arrow::read_parquet("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/land_cover/deforestation_municipalities_upstream.parquet")
land_cover <- land_cover %>% mutate(across(municipality, as.double))

lm(I(deforestation / total) ~ I(cloud_cover / 100) * I(year > 2003), data = land_cover) %>% summary()

analysis <- inner_join(mortality, land_cover, by = c("mun_id" = "municipality", "year")) %>% drop_na(deforestation)


analysis <- analysis %>% 
    group_by(mun_id) %>%
    mutate(across(c(deforestation, total), cumsum),
           across(c(cloud_cover), cummean),
           forest_rate = forest / total,
           deforestation_rate = deforestation / total) %>%
    ungroup() %>%
    filter(mortality_rate != Inf) %>%
    drop_na(deforestation_rate, mortality_rate)

# panel details
analysis

test <- analysis %>% filter(year >= 2003)
lm(deforestation_rate ~ I(cloud_cover / 100), data = test) %>% summary()
lm(forest_rate ~ I(cloud_cover / 100), data = test) %>% summary()

felm(mortality_rate ~ 1 | mun_id + year | (deforestation_rate ~ cloud_cover) | mun_id + year, data = test) %>% summary()



###
#
###

deforestation <- arrow::read_parquet("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/land_cover/deforestation_municipalities.parquet")

deforestation <- deforestation %>% 
    arrange(municipality, year) %>%
    group_by(municipality) %>%
    mutate(cloud_cover_ll_1 = lag(cloud_cover, 1),
           cloud_cover_ll_2 = lag(cloud_cover, 2)) %>%
    ungroup()

test <- deforestation %>% filter(year >= 2003)

felm(I(deforestation / total) ~ cloud_cover + cloud_cover_ll_1 + cloud_cover_ll_2 | municipality + year | 0 | municipality + year, data = test) %>% summary()


###
#
###

setwd("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis")

library(tidyverse)
library(splm)
library(plm)
library(spdep)

# read data
mortality <- arrow::read_parquet("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/mortality/mortality_panel.parquet")

# child mortality
mortality <- mortality %>% 
    drop_na(deaths, population) %>% 
    filter(age_group %in% c("under_1", "1_to_4")) %>%
    group_by(mun_id, year) %>%
    summarise(mortality_rate = sum(deaths) / sum(population)) %>%
    ungroup() %>%
    mutate(municipality = as.character(mun_id))


deforestation <- arrow::read_parquet("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/land_cover/deforestation_municipalities.parquet")

deforestation <- deforestation %>% 
    arrange(municipality, year) %>%
    group_by(municipality) %>%
    mutate(cloud_cover_ll_1 = lag(cloud_cover, 1),
           cloud_cover_ll_2 = lag(cloud_cover, 2)) %>%
    ungroup()

analysis <- inner_join(mortality, deforestation, by = c("municipality", "year")) %>% drop_na(deforestation)

analysis <- analysis %>% 
    group_by(mun_id) %>%
    mutate(across(c(deforestation, total), cumsum),
           across(c(cloud_cover), cummean),
           forest_rate = forest / total,
           deforestation_rate = deforestation / total) %>%
    ungroup() %>%
    filter(mortality_rate != Inf) %>%
    drop_na(deforestation_rate, mortality_rate)

# Count the number of observations for each municipality and year
obs_count <- analysis %>%
  group_by(municipality, year) %>%
  summarise(count = n()) %>%
  ungroup()

# Check which municipalities have observations for each year
year_count <- obs_count %>%
  group_by(municipality) %>%
  summarise(years_observed = n()) %>%
  ungroup()

# Identify the largest balanced sub-panel
max_years <- max(year_count$years_observed)
balanced_municipalities <- year_count %>%
  filter(years_observed == max_years) %>%
  pull(municipality)

# Filter the data to include only the balanced sub-panel
balanced_analysis <- analysis %>%
  filter(municipality %in% balanced_municipalities)

# Check the balance of the resulting data
balanced_check <- balanced_analysis %>%
  group_by(municipality, year) %>%
  summarise(count = n()) %>%
  ungroup() %>%
  spread(key = year, value = count)

# If any NA values exist, it means the panel is not balanced
if(any(is.na(balanced_check))) {
  print("The panel is not balanced")
} else {
  print("The panel is balanced")
}


subset_analysis <- balanced_analysis %>% 
    #filter(mun_id %in% c(110001, 110002)) %>%
    select(mun_id, year, deforestation_rate, mortality_rate, cloud_cover_ll_1)

lfe::felm(deforestation_rate ~ cloud_cover | mun_id + year, data = balanced_analysis %>% filter(year >= 2003)) %>% summary()

data_municipalities <- as.double(unique(subset_analysis$mun_id))

weights_municipalities <- read_csv("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/river_network/weights_municipalities.csv")
weights_municipalities <- weights_municipalities %>% 
    mutate(in_panel = map_lgl(cc_2r, function(x) x %in% data_municipalities))

prepare_weights <- function(method, max_distance, weights_municipalities){

    weights <- Matrix::readMM(paste0("data/river_network/weights_matrix_", method, "_", max_distance, ".mtx"))
    weights <- as.matrix(weights[weights_municipalities$in_panel, weights_municipalities$in_panel])

    rownames(weights) <- weights_municipalities$cc_2r[weights_municipalities$in_panel]
    colnames(weights) <- weights_municipalities$cc_2r[weights_municipalities$in_panel]

    # Identify disconnected units
    disconnected_units <- which(rowSums(weights) == 0)

    # introduce small back-weights at river network end nodes
    to_remove <- c()
    for(i in disconnected_units){
        if(!colSums(weights)[i] == 0) {
            for(j in which(weights[, i] > 0)){
                weights[i,j] <- .1
            }
        } else {
            to_remove <- c(to_remove, i)
        }
    }

    # Remove completely disconnected units from the weights matrix
    if (length(to_remove) > 0) {
    weights_matrix_filtered <- weights[-to_remove, -to_remove]
    } else {
    weights_matrix_filtered <- weights
    }

    # Convert the filtered weights matrix back to listw format if needed
    #weights <- mat2listw(weights, style = "W", zero.policy = TRUE)
    weights_filtered <- mat2listw(weights_matrix_filtered, style = "W")

    return(list(weights_filtered = weights_filtered, weights_matrix_filtered = weights_matrix_filtered))
}

# Function to calculate the two-way cluster-robust variance-covariance matrix with HC0 adjustment
two_way_cluster_hc0_vcov <- function(X, residuals, cluster_id, cluster_time) {
  # Compute bread matrix (X'X)^(-1)
  bread <- solve(t(X) %*% X)
  
  # Compute meat matrix using unadjusted residuals
  X_u <- X * residuals
  
  # Create cluster sums
  cluster_sum_id <- rowsum(X_u, cluster_id)
  cluster_sum_time <- rowsum(X_u, cluster_time)
  
  # Meat components
  meat_id <- t(cluster_sum_id) %*% cluster_sum_id
  meat_time <- t(cluster_sum_time) %*% cluster_sum_time
  
  # Adjust for double-counting
  meat <- (meat_id + meat_time - (t(X_u) %*% X_u))
  
  # Small sample correction factors
  G1 <- length(unique(cluster_id))
  G2 <- length(unique(cluster_time))
  N <- nrow(X)
  df_correction <- (G1 * G2) / (G1 + G2 - 1)
  
  # Apply small sample correction
  meat_corrected <- df_correction * meat
  
  # Variance-covariance matrix
  vcov <- bread %*% meat_corrected %*% bread
  return(vcov)
}


# Function to calculate the two-way cluster-robust variance-covariance matrix with HC1 adjustment
two_way_cluster_hc1_vcov <- function(X, residuals, cluster_id, cluster_time) {
  # Compute bread matrix (X'X)^(-1)
  bread <- solve(t(X) %*% X)
  
  # Apply HC1 adjustment to residuals
  N <- nrow(X)
  K <- ncol(X)
  hc1_factor <- sqrt(N / (N - K))
  hc1_residuals <- residuals * hc1_factor
  
  # Compute meat matrix using HC1 adjusted residuals
  X_hc1_u <- X * hc1_residuals
  
  # Create cluster sums
  cluster_sum_id <- rowsum(X_hc1_u, cluster_id)
  cluster_sum_time <- rowsum(X_hc1_u, cluster_time)
  
  # Meat components
  meat_id <- t(cluster_sum_id) %*% cluster_sum_id
  meat_time <- t(cluster_sum_time) %*% cluster_sum_time
  
  # Adjust for double-counting
  meat <- (meat_id + meat_time - (t(X_hc1_u) %*% X_hc1_u))
  
  # Small sample correction factors
  G1 <- length(unique(cluster_id))
  G2 <- length(unique(cluster_time))
  df_correction <- (G1 * G2) / (G1 + G2 - 1)
  
  # Apply small sample correction
  meat_corrected <- df_correction * meat
  
  # Variance-covariance matrix
  vcov <- bread %*% meat_corrected %*% bread
  return(vcov)
}

estimate_models <- function(subset_analysis, weights_filtered, weights_matrix_filtered){
    subset_pdata_filtered <- pdata.frame(subset_analysis %>% filter(mun_id %in% rownames(weights_matrix_filtered)) %>% arrange(mun_id, year), index = c("mun_id", "year"))

    # First stage: Regress the endogenous variable on the instruments and other exogenous variables
    iv_model <- plm(deforestation_rate ~ cloud_cover_ll_1, data = subset_pdata_filtered, model = "within", effect = "twoways")

    # Get the fitted values (instrumented values for x1)
    subset_pdata_filtered$x1_hat <- fitted(iv_model)

    # Second stage: Estimate the SDM with the instrumented variable
    # Create the formula with the spatial lag of the dependent variable and independent variables
    formula <- mortality_rate ~ x1_hat

    # Estimate the SDM with two-way fixed effects
    sdm_model <- spml(formula, data = subset_pdata_filtered, listw = weights_filtered, model = "within", effect = "twoways", spatial.error = "none", lag = TRUE)

    # Extract the model matrix and residuals
    X <- model.matrix(formula, data = subset_pdata_filtered)
    residuals <- residuals(sdm_model)
    n <- nrow(X)
    k <- ncol(X)

    # Define the clustering variables
    cluster_id <- as.numeric(subset_pdata_filtered$mun_id)
    cluster_time <- as.numeric(subset_pdata_filtered$year)

    # Calculate the two-way cluster-robust variance-covariance matrix
    vcov_two_way <- two_way_cluster_hc1_vcov(X, residuals, cluster_id, cluster_time)

    # Get the robust standard errors
    robust_se <- sqrt(diag(vcov_two_way))

    return(list(iv_model = iv_model, sdm_model = sdm_model, robust_se = robust_se))
}

for (max_distance in c(100, 500, 1000)){
    for (method in c("linear", "exponential")){
        tmp <- prepare_weights(method, max_distance, weights_municipalities)
        weights_filtered <- tmp$weights_filtered; weights_matrix_filtered <- tmp$weights_matrix_filtered

        results <- estimate_models(subset_analysis, weights_filtered, weights_matrix_filtered)
        saveRDS(results, file = paste0("output/models/spml_deforestationTotal_noControl_", method, "_", max_distance, ".rds"))
    }
}


results <- readRDS("output/models/spml_deforestationTotal_noControl_linear_100.rds")
results$"sdm_model" %>% summary()

# Get the robust standard errors
robust_se <- sqrt(diag(vcov_two_way))

# Summarize the results with robust standard errors
coef <- coefficients(sdm_model)
summary <- data.frame(Estimate = coef, `Robust SE` = robust_se, `t value` = coef / robust_se, `p` = 2 * pt(-abs(coef / robust_se), df = n - k))

print(summary)

###
# Plot weights
###

library(tidyverse)
library(sf)
library(Matrix)

municipalities <- st_read("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/boundaries/gadm41_BRA_2.json")

municipalities <- municipalities %>% mutate(CC_2r = str_sub(CC_2, 1, 6))

weights_municipalities <- read_csv("/Users/felixschulz/Downloads/weights_municipalities.csv")
    
c_cc2r <- municipalities %>% select(CC_2r) %>% unique() %>% pluck(1, 2000)

distances <- readMM("/Users/felixschulz/Downloads/distance_matrix.mtx")
rownames(distances) <- weights_municipalities$cc_2r
colnames(distances) <- weights_municipalities$cc_2r

c_muns_plot <- inner_join(municipalities, enframe(distances[c_cc2r,], name = "CC_2r", value = "distance")) %>%
    filter(distance > 0)

ggplot() +
    geom_sf(aes(fill = distance), data = c_muns_plot) +
    geom_sf(data = c_mun, fill = "red")

weights <- readMM("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/river_network/weights_matrix_linear_1000.mtx")
rownames(weights) <- weights_municipalities$cc_2r
colnames(weights) <- weights_municipalities$cc_2r

c_muns_plot <- inner_join(municipalities, enframe(weights[c_cc2r,], name = "CC_2r", value = "weight")) %>%
    filter(weight > 0)

ggplot() +
    geom_sf(data = c_mun, fill = "red") +
    geom_sf(aes(fill = weight), data = c_muns_plot)


###
# Plot mortality
###

# load data and prepare for plotting
mortality <- arrow::read_parquet("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/mortality/mortality_panel.parquet")

mortality <- mortality %>% 
    drop_na(deaths, population) %>% 
    filter(age_group %in% c("under_1", "1_to_4")) %>%
    group_by(mun_id, year) %>%
    summarise(mortality_rate = sum(deaths) / sum(population)) %>%
    ungroup()

mortality <- mortality %>%
    group_by(mun_id) %>%
    filter(!all(mortality_rate == 0)) %>%
    ungroup()

# plot mortality rates

# get examples of municipalities with low, medium, and high mortality rates
mortality %>%
    group_by(mun_id) %>%
    summarise(mortality_rate = mean(mortality_rate)) %>%
    ungroup() %>%
    arrange(mortality_rate) %>%
    slice(c(1, round(n() / 2), n())) %>%
    pull(mun_id) %>%
    set_names(c("low", "medium", "high")) -> example_municipalities


ggplot(mortality %>% filter(mun_id %in% example_municipalities)) +
    geom_point(aes(x = year, y = mortality_rate, color = as.character(mun_id)))

# check for outliers within municipalities (outside 1.5 IQR)
mortality_outliers <- mortality %>%
    drop_na(mortality_rate) %>%
    group_by(municipality) %>%
    mutate(is_outlier = mortality_rate > median(mortality_rate) + 1.5 * IQR(mortality_rate) | mortality_rate < median(mortality_rate) - 1.5 * IQR(mortality_rate)) %>%
    ungroup()


mortality_outliers$is_outlier %>% summary()
