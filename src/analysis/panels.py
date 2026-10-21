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
    """Load the municipality x year panel, restricted to `years` (None keeps all).

    `has_upstream` is False for headwater municipalities, which have local
    (`local_*`) but no upstream (`lc_*`/`alr_*`, upstream climate) exposure.
    """
    frame = pd.read_parquet(path)
    frame["has_upstream"] = frame["lc_forest"].notna()
    return _restrict_years(frame, years).reset_index(drop=True)


# --- Health outcomes ---------------------------------------------------------
# SIH hospitalizations and SINASC births are joined into `adm2_panel` by
# assembly; these helpers name their column groups from the config.

# SIH legacy (<= 2007) and current (>= 2008) series never overlap in a year.
SIH_LAST_LEGACY_YEAR = 2007


def hospitalization_columns(breakdown=None):
    """Return the `adm2_panel` hospitalization columns for one breakdown.

    `None` gives the all-cause metrics (`hosp_*_count`), `"icd10_chapter"`
    the `hosp_icd_*` counts and `"morbidity_channel"` the channel counts.
    """
    sources = {
        None: "hospitalizations",
        "icd10_chapter": "hospitalizations_icd10_chapter",
        "morbidity_channel": "hospitalizations_morbidity_channel",
    }
    if breakdown not in sources:
        raise ValueError(f"Unknown breakdown {breakdown!r}.")
    return _assembly_source_variables("adm2_panel", sources[breakdown])


def birth_columns():
    """Return the `adm2_panel` SINASC birth-count columns."""
    return _assembly_source_variables("adm2_panel", "births")


def sih_source_system(year):
    """Label each year's SIH series: `"legacy"` up to 2007, `"current"` from 2008."""
    return pd.Series(year).le(SIH_LAST_LEGACY_YEAR).map({True: "legacy", False: "current"}).to_numpy()
