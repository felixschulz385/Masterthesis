#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| include: false
library("tidyverse")
library("arrow")
library("sf")
library("rmapshaper")
library("stargazer")

r0 = function(x) round(x, 0)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| label: "fig-mortality-rates"
#| fig-cap: "Sterberate Altergruppe <5 in Brasilien, 2015"
child_mortality_data = read_parquet("data/brazil_mortality/mortality_data_u5_2015.parquet")
map_brazil = read_sf("data/maps/brazil_municipalities_simplified_map.json")

merged = child_mortality_data %>% 
    left_join(map_brazil, by = c("CODMUNRES" = "CC_2")) %>%
    drop_na() %>%
    st_sf()

ggplot() +
  geom_sf(data = merged, aes(fill = mortality_rate), color = NA) +
  scale_fill_viridis_c(option = "magma", direction = -1) +
  #ggtitle("Mortality rate of children under the age of 5 in 2015") +
  theme_bw()
#
#
#
#
#
#| echo: false
#| output: asis
#| tbl-cap: "Politische Orientierung und Sterberate Altersgruppe unter 5, Brasilien, 2015"
vote = read_parquet("data/brazil_vote/resultados_candidato_municipio.parquet")

share_pl = vote %>% 
    filter(ano == 2022, turno == 2) %>%
    group_by(id_municipio = str_sub(id_municipio, 1, 6)) %>%
    summarise(share_pl = sum(votos[sigla_partido == "PL"]) / sum(votos))

merged = share_pl %>%
    left_join(map_brazil, by = c("id_municipio" = "CC_2")) %>%
    left_join(child_mortality_data, by = c("id_municipio" = "CODMUNRES")) %>%
    st_sf()

lm(mortality_rate ~ share_pl + 1, data = merged) %>% stargazer(header = F)
#
#
#
#
