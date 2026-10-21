"""Loaders and variable groups for the assembled analysis panels.

Shared by the notebooks under `output/notebooks/` so that the pollution
(sensor-indexed) and health (ADM2-indexed) workflows read the same columns
the same way. Variable groups are derived from `setup/assembly_datasets.yaml`
where possible rather than hard-coded, so they track the assembly config.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .sensor_data.catalog import _classify_pollutant_type
from .settings import PROJECT_ROOT


ASSEMBLY_CONFIG_PATH = PROJECT_ROOT / "setup" / "assembly_datasets.yaml"
SENSOR_PANEL_PATH = PROJECT_ROOT / "data" / "assembly" / "sensor_panel.parquet"
ADM2_PANEL_PATH = PROJECT_ROOT / "data" / "assembly" / "adm2_panel.parquet"
HEALTH_PROCESSED_DIR = PROJECT_ROOT / "data" / "health" / "processed"

# Years covered by both the land-cover (MapBiomas) and climate (ERA5-Land)
# upstream aggregates; outside this window the exposure/controls are missing.
EXPOSURE_YEARS = (1985, 2024)

LAND_COVER_SHARE_COLUMNS = [
    "lc_forest",
    "lc_nonforest_nat",
    "lc_pasture",
    "lc_agriculture",
    "lc_farming_unclassified",
    "lc_urban",
    "lc_mining",
    "lc_other",
    "lc_water",
]
LAND_COVER_ALR_COLUMNS = [
    "alr_pasture",
    "alr_agriculture",
    "alr_farming_unclassified",
    "alr_urban",
    "alr_mining",
    "alr_other",
]

CLIMATE_VARIABLES = ["2t", "2d", "tp", "sro", "ssro", "pev", "swvl1", "swvl2", "2t_daily_min", "2t_daily_max"]
SENSOR_CLIMATE_WINDOWS = ["day", "7d", "30d", "90d", "180d", "365d"]

STREAMFLOW_COLUMNS = [
    "streamflow_discharge_day",
    "streamflow_discharge_mean_7d",
    "streamflow_discharge_mean_31d",
]

# Columns of the water_quality_streamflow source that are sampling metadata
# or covariates rather than water-quality analytes.
SENSOR_NON_ANALYTE_COLUMNS = {
    "rained",
    "depth",
    "depth_parameter",
    "air_temperature",
    "trench_id",
    "streamflow_match_count",
    "streamflow_nonnull_day_count",
    "streamflow_total_weight",
    "streamflow_nearest_distance_m",
    *STREAMFLOW_COLUMNS,
}


def sensor_climate_columns(variables=None, windows=None):
    """Return `{variable}_mean_{window}` sensor-panel climate column names."""
    return [
        f"{variable}_mean_{window}"
        for variable in (variables or CLIMATE_VARIABLES)
        for window in (windows or SENSOR_CLIMATE_WINDOWS)
    ]


def _assembly_source_variables(dataset_id, source_name):
    with open(ASSEMBLY_CONFIG_PATH) as config_file:
        config = yaml.safe_load(config_file)
    for dataset in config["datasets"]:
        if dataset["id"] != dataset_id:
            continue
        for source in dataset["sources"]:
            if source["name"] == source_name:
                return list(source["variables"])
    raise KeyError(f"No source '{source_name}' in assembly dataset '{dataset_id}'.")


def sensor_analyte_columns():
    """Return the water-quality analyte columns declared for the sensor panel."""
    variables = _assembly_source_variables("sensor_panel", "water_quality_streamflow")
    return [variable for variable in variables if variable not in SENSOR_NON_ANALYTE_COLUMNS]


def analyte_type_groups(analytes):
    """Map each analyte to the pollutant type group used by the sensor_data pipeline."""
    return {analyte: _classify_pollutant_type(analyte) for analyte in analytes}


def _restrict_years(frame, years):
    if years is None:
        return frame
    first_year, last_year = years
    return frame[frame["year"].between(first_year, last_year)]


def load_sensor_panel(path=SENSOR_PANEL_PATH, years=EXPOSURE_YEARS):
    """Load the sensor x sample-datetime panel, restricted to `years` (None keeps all)."""
    frame = pd.read_parquet(path)
    frame = frame.dropna(subset=["datetime"])
    frame["year"] = frame["year"].astype(int)
    frame = _restrict_years(frame, years)
    frame["month"] = frame["datetime"].dt.month
    return frame.reset_index(drop=True)


def load_adm2_panel(path=ADM2_PANEL_PATH, years=EXPOSURE_YEARS):
    """Load the municipality x year panel, restricted to `years` (None keeps all)."""
    frame = pd.read_parquet(path)
    return _restrict_years(frame, years).reset_index(drop=True)


# --- Health outcomes ---------------------------------------------------------
# Health is not yet wired into `adm2_panel` (see the "Deferred" note at the end
# of setup/assembly_datasets.yaml). Until it is, the helpers below pivot the
# processed health tables into wide (mun_id, year) frames so the health
# descriptives can run; they should be replaced by assembly once health is
# added there.

HOSPITALIZATION_METRICS = ["hospitalizations_count", "in_hospital_deaths_count", "inpatient_days_count"]


def load_hospitalizations(metric="hospitalizations_count", breakdown=None):
    """Return SIH (residence-based) hospitalizations wide by (mun_id, year).

    `breakdown=None` gives all-cause totals for `HOSPITALIZATION_METRICS`;
    `"icd10_chapter"` or `"morbidity_channel"` gives one column per category
    for the single `metric`. A `source_system` column records the SIH legacy
    (<=2007) vs current (>=2008) series, which overlap in no year.
    """
    if breakdown is None:
        path = HEALTH_PROCESSED_DIR / "health_hospitalizations.parquet"
        frame = pd.read_parquet(path)
        frame = frame[frame["metric_name"].isin(HOSPITALIZATION_METRICS)]
        pivot_column = "metric_name"
        prefix = "hosp_"
    elif breakdown == "icd10_chapter":
        path = HEALTH_PROCESSED_DIR / "health_hospitalizations_icd10_chapter.parquet"
        frame = pd.read_parquet(path, filters=[("metric_name", "==", metric)])
        pivot_column = "icd10_chapter_code"
        prefix = "hosp_icd_"
    elif breakdown == "morbidity_channel":
        path = HEALTH_PROCESSED_DIR / "health_hospitalizations_selected_morbidity_list.parquet"
        frame = pd.read_parquet(path, filters=[("metric_name", "==", metric)])
        pivot_column = "morbidity_channel"
        prefix = "hosp_"
    else:
        raise ValueError(f"Unknown breakdown {breakdown!r}.")

    frame = frame.rename(columns={"municipality_code": "mun_id"})
    wide = frame.pivot_table(
        index=["mun_id", "year", "source_system"],
        columns=pivot_column,
        values="metric_value",
        aggfunc="sum",
    )
    wide.columns = [f"{prefix}{column}" for column in wide.columns]
    return wide.reset_index()


def load_births():
    """Return SINASC births with low-birth-weight and preterm counts by (mun_id, year)."""
    weight = pd.read_parquet(HEALTH_PROCESSED_DIR / "health_birth_weight.parquet")
    gestation = pd.read_parquet(HEALTH_PROCESSED_DIR / "health_gestational_duration.parquet")

    low_weight_bins = ["Menos de 500g", "500 a 999g", "1000 a 1499 g", "1500 a 2499 g"]
    births = weight[["mun_id", "year"]].copy()
    births["births"] = weight["Total"]
    births["births_weight_known"] = weight["Total"] - weight["Ignorado"]
    births["births_low_weight"] = weight[low_weight_bins].sum(axis=1)

    # 28-36 weeks is reported either as one bin or split into 28-31/32-36
    # depending on the year; summing with min_count=0 covers both layouts.
    preterm_bins = [
        "Menos de 22 semanas",
        "De 22 a 27 semanas",
        "De 28 a 36 semanas, não especificado",
        "De 28 a 31 semanas",
        "De 32 a 36 semanas",
    ]
    preterm = gestation[["mun_id", "year"]].copy()
    preterm["births_gestation_known"] = gestation["Total"] - gestation["Ignorado"]
    preterm["births_preterm"] = gestation[preterm_bins].sum(axis=1, min_count=0)
    return births.merge(preterm, on=["mun_id", "year"], how="outer", validate="one_to_one")
