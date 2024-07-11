library(tidyverse)
library(arrow)
# library(splm)
# library(plm)
# library(spdep)
library(lfe)
library(texreg)

setwd("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis")

###
# Load data
###

large_panel <- read_parquet("data/analysis/large_panel.parquet")
small_panel <- read_parquet("data/analysis/small_panel.parquet")
large_panel_l1 <- read_parquet("data/analysis/large_panel_l1.parquet")
small_panel_l1 <- read_parquet("data/analysis/small_panel_l1.parquet")

###
# First stage, Deter
###

first_stage_deter <- list()

first_stage_deter[["0c"]] <- felm(deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream | municipality + year | 0 | municipality + region_year, data = large_panel)
first_stage_deter[["1c"]] <- felm(deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream + temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel)
first_stage_deter[["2c"]] <- felm(deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream + temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | 0 | municipality + region_year, data = large_panel)
first_stage_deter[["3c"]] <- felm(deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream + temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | 0 | municipality + region_year, data = small_panel)

#c("Cloud Cover", "Temperature", "Precipitation", "GDP per Capita", "Vaccination Index", "Doctors per 1000", "Education", "Primary Care Coverage")
texreg(first_stage_deter,
    file = "output/tables/reg_deforestation_DETER_first_stage.tex",
    custom.header = list("Deforestation" = 1:4),
    custom.model.names = c("A", "B", "C", "D"),
    custom.coef.names = c("Cloud Cover", "", "", "", "", "", "", ""),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "Regression Results: First stage results",
    label = "tbl-reg-deforestation-sensors",
    include.rsquared = FALSE, include.adjrs = FALSE, 
    custom.gof.rows = list(
        "Climate Controls" = list("", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "", "\\ding{51}"),
        "first-stage F Statistic" = map_dbl(first_stage_deter, ~ .x %>% summary() %>% pluck("P.fstat") %>% pluck("F"))
    ),
    custom.note = "\\item \\textit{Notes:} I regress the cumulative, weighted upstream deforestation rate on the average, weighted upstream DETER cloud coverage. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls are gdp per capita, educational scores, and a vaccination index for waterborne diseases. Full Controls add the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    #scalebox = 1,
    dcolumn = TRUE
)

###
# Second stage, Deter, Total Mortality
###

second_stage_deter_mortality_tot <- list()
second_stage_deter_mortality_tot[["0c"]] <- felm(mortality_rate_tot ~ 1 | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_mortality_tot[["1c"]] <- felm(mortality_rate_tot ~ temperature + precipitation | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_mortality_tot[["2c"]] <- felm(mortality_rate_tot ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_mortality_tot[["3c"]] <- felm(mortality_rate_tot ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)

texreg(
    second_stage_deter_mortality_tot,
    file = "output/tables/reg_deforestation_DETER_mortality_tot.tex",
    custom.header = list("Mortality Rate" = 1:4),
    custom.model.names = c("A", "B", "C", "D"),
    custom.coef.names = c("Deforestation", "", "", "", "", "", "", ""),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption =  "Regression Results: Upstream Deforestation and Mortality (per 1,000 inhabitants)",
    label = "tbl-reg-deforestation-sensors",
    include.rsquared = FALSE, include.adjrs = FALSE, 
    custom.gof.rows = list(
        "Climate Controls" = list("", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the overall mortality rate on the instrumented weighted upstream deforestation rate. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls are gdp per capita, educational scores, and a vaccination index for waterborne diseases. Full Controls add the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .8,
    dcolumn = TRUE
)

###
# Second stage, Deter, Infant Mortality
###

second_stage_deter_mortality_l1 <- list()
second_stage_deter_mortality_l1[["0c"]] <- felm(mortality_rate_l1 ~ 1 | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel_l1)
second_stage_deter_mortality_l1[["1c"]] <- felm(mortality_rate_l1 ~ temperature + precipitation | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel_l1)
second_stage_deter_mortality_l1[["2c"]] <- felm(mortality_rate_l1 ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel_l1)
second_stage_deter_mortality_l1[["3c"]] <- felm(mortality_rate_l1 ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel_l1)

texreg(
    second_stage_deter_mortality_l1,
    file = "output/tables/reg_deforestation_DETER_mortality_l1.tex",
    custom.header = list("Infant Mortality Rate" = 1:4),
    custom.model.names = c("A", "B", "C", "D"),
    custom.coef.names = c("Deforestation", "", "", "", "", "", "", ""),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "Regression Results: Upstream Deforestation and Infant Mortality (per 1,000 births)",
    label = "tbl-reg-deforestation-sensors",
    include.rsquared = FALSE, include.adjrs = FALSE, 
    custom.gof.rows = list(
        "Climate Controls" = list("", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "", "\\ding{51}"),
        "first-stage F Statistic" = map_dbl(first_stage_deter, ~ .x %>% summary() %>% pluck("P.fstat") %>% pluck("F"))
    ),
    custom.note = "\\item \\textit{Notes:} I regress the infant mortality rate on the instrumented weighted upstream deforestation rate. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls are gdp per capita, educational scores, and a vaccination index for waterborne diseases. Full Controls add the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    #scalebox = 1,
    dcolumn = TRUE
)

###
# Second stage, Deter, Hospitalizations
###

second_stage_deter_hosp_rate <- list()
second_stage_deter_hosp_rate[["0c"]] <- felm(hosp_rate ~ 1 | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)
second_stage_deter_hosp_rate[["1c"]] <- felm(hosp_rate ~ temperature + precipitation | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)
second_stage_deter_hosp_rate[["2c"]] <- felm(hosp_rate ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)
second_stage_deter_hosp_rate[["3c"]] <- felm(hosp_rate ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)

texreg(
    second_stage_deter_hosp_rate,
    file = "output/tables/reg_deforestation_DETER_hosp_rate.tex",
    custom.header = list("Medical Expenditure per Inhabitant" = 1:4),
    custom.model.names = c("A", "B", "C", "D"),
    custom.coef.names = c("Deforestation", "", "", "", "", "", "", ""),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "Regression Results: Upstream Deforestation and Hospitalizations (per 1000 inhabitants)",
    label = "tbl-reg-deforestation-sensors",
    include.rsquared = FALSE, include.adjrs = FALSE, 
    custom.gof.rows = list(
        "Climate Controls" = list("", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the hospitalization rate on the instrumented weighted upstream deforestation rate. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls are gdp per capita, educational scores, and a vaccination index for waterborne diseases. Full Controls add the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    #scalebox = 1,
    dcolumn = TRUE
)

###
# Second stage, Deter, Expenditure per Inhabitant
###

second_stage_deter_ex_pop <- list()
second_stage_deter_ex_pop[["0c"]] <- felm(ex_pop ~ 1 | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)
second_stage_deter_ex_pop[["1c"]] <- felm(ex_pop ~ temperature + precipitation | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)
second_stage_deter_ex_pop[["2c"]] <- felm(ex_pop ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)
second_stage_deter_ex_pop[["3c"]] <- felm(ex_pop ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (deforestation_cum_r2005_upstream_rate ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)

texreg(
    second_stage_deter,
    file = "output/tables/reg_deforestation_DETER_ex_pop.tex",
    custom.header = list("Medical Expenditure per Inhabitant" = 1:4),
    custom.model.names = c("A", "B", "C", "D"),
    custom.coef.names = c("Deforestation", "", "", "", "", "", "", ""),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "Regression Results: Upstream Deforestation and Medical Expenditure per Inhabitant",
    label = "tbl-reg-deforestation-sensors",
    include.rsquared = FALSE, include.adjrs = FALSE, 
    custom.gof.rows = list(
        "Climate Controls" = list("", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the medical expenditure rate on the instrumented weighted upstream deforestation rate. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls are gdp per capita, educational scores, and a vaccination index for waterborne diseases. Full Controls add the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    #scalebox = 1,
    dcolumn = TRUE
)

# OLS

ols_deter <- list()

ols_deter[["Total Mortality"]] <- felm(mortality_rate_tot ~ deforestation_cum_r2005_upstream_rate + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | 0 | municipality + region_year, data = large_panel)
ols_deter[["Infant Mortality"]] <- felm(mortality_rate_l1 ~ deforestation_cum_r2005_upstream_rate + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | 0 | municipality + region_year, data = large_panel_l1)
ols_deter[["Hospitalization Rate"]] <- felm(hosp_rate ~ deforestation_cum_r2005_upstream_rate + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | 0 | municipality + region_year, data = small_panel)
ols_deter[["Expenditure per Inhabitant"]] <- felm(ex_pop ~ deforestation_cum_r2005_upstream_rate + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | 0 | municipality + region_year, data = small_panel)

texreg(
    ols_deter,
    file = "output/tables/reg_deforestation_ols.tex",
    #custom.model.names = c("1a", "1b", "1c", "1d"),
    custom.coef.names = c("Deforestation", "", "", "", ""),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "Regression Results: Upstream Deforestation and Medical Expenditure per Inhabitant",
    label = "tbl-reg-deforestation-sensors",
    include.rsquared = FALSE, include.adjrs = FALSE, 
    custom.gof.rows = list(
        "Full Controls" = list("\\ding{51}", "\\ding{51}", "\\ding{51}", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the medical expenditure rate on the instrumented weighted upstream deforestation rate. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls are gdp per capita, educational scores, and a vaccination index for waterborne diseases. Full Controls add the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    #scalebox = 1,
    dcolumn = TRUE
)
