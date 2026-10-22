"""Otto Pfafstetter sub-basin codes per trench.

ANA's BHO drainage-area layer carries each trench's Otto basin code at levels
1-6 (`nunivotto1` ... `nunivotto6`; level k is the first k digits of
`cobacia`). `river_network`'s connected-component `system_id` is too coarse
for clustering (one system holds ~42% of sensor samples), so these nested
basins are kept as a separate, attribute-only output rather than a column of
`river_network_trenches.parquet`: building it needs no geometry and no rerun of
the full network/matrix pipeline.

`river_network_adm2_otto_basins.parquet` assigns each ADM2 region its dominant
basin per level (largest summed drainage area of its intersecting trenches),
the clustering unit for the municipality-indexed health panel.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .constants import (
    ADM2_COLUMN,
    ADM2_OTTO_BASINS_FILENAME,
    DEFAULT_RAW_GPKG_DRAINAGE_LAYER,
    DRAINAGE_AREAS_FILENAME,
    OTTO_BASINS_FILENAME,
    TRENCH_ADM2_TABLE_FILENAME,
    TRENCH_ID_COLUMN,
    TRENCHES_FILENAME,
)


logger = logging.getLogger(__name__)

OTTO_BASIN_LEVELS = (1, 2, 3, 4, 5, 6)


def otto_basin_column(level):
    """Output column name for the Otto basin code at `level`."""
    return f"otto_basin_l{level}"


OTTO_BASIN_COLUMNS = [otto_basin_column(level) for level in OTTO_BASIN_LEVELS]
DRAINAGE_AREA_COLUMN = "drainage_area"


def build_otto_basin_table(gpkg_path, layer=DEFAULT_RAW_GPKG_DRAINAGE_LAYER, trench_ids=None):
    """Read `trench_id` plus Otto basin codes (string, levels 1-6) from the BHO drainage layer.

    Duplicate trench rows are dropped keeping the first, matching how
    `RiverNetwork` deduplicates drainage areas (the 31 duplicates in BHO 2017
    v01.05 carry identical codes). Pass `trench_ids` to restrict the table to
    the trenches of a (possibly bbox-subset) network.
    """
    import pyogrio

    source_columns = ["cotrecho", *(f"nunivotto{level}" for level in OTTO_BASIN_LEVELS)]
    logger.info("Reading Otto basin codes from %s layer %s", gpkg_path, layer)
    table = pyogrio.read_dataframe(gpkg_path, layer=layer, columns=source_columns, read_geometry=False)
    table = table.rename(
        columns={
            "cotrecho": TRENCH_ID_COLUMN,
            **{f"nunivotto{level}": otto_basin_column(level) for level in OTTO_BASIN_LEVELS},
        }
    )
    table = table.drop_duplicates(subset=TRENCH_ID_COLUMN, keep="first")
    if trench_ids is not None:
        table = table[table[TRENCH_ID_COLUMN].isin(pd.Index(trench_ids))]
    table[OTTO_BASIN_COLUMNS] = table[OTTO_BASIN_COLUMNS].astype("string")
    logger.info("Built Otto basin table for %d trenches", len(table))
    return table.reset_index(drop=True)


def build_adm2_otto_basin_table(trench_adm2, otto_basins, drainage_areas):
    """Assign each ADM2 region its dominant Otto basin at every level.

    A basin's weight within an ADM2 is the summed `drainage_area` of the
    ADM2's intersecting trenches (`trench_adm2`, one row per trench x ADM2) in
    that basin. Levels are resolved independently, so the level-2 code need
    not be a prefix of the level-3 one when a region straddles basins; ties go
    to the smaller code.
    """
    merged = trench_adm2[[TRENCH_ID_COLUMN, ADM2_COLUMN]].merge(
        otto_basins[[TRENCH_ID_COLUMN, *OTTO_BASIN_COLUMNS]], on=TRENCH_ID_COLUMN, how="inner"
    ).merge(
        drainage_areas[[TRENCH_ID_COLUMN, DRAINAGE_AREA_COLUMN]].drop_duplicates(TRENCH_ID_COLUMN),
        on=TRENCH_ID_COLUMN,
        how="inner",
    )
    table = pd.DataFrame({ADM2_COLUMN: pd.Series(sorted(merged[ADM2_COLUMN].unique()), dtype=merged[ADM2_COLUMN].dtype)})
    for column in OTTO_BASIN_COLUMNS:
        dominant = (
            merged.groupby([ADM2_COLUMN, column], as_index=False)[DRAINAGE_AREA_COLUMN]
            .sum()
            .sort_values([ADM2_COLUMN, DRAINAGE_AREA_COLUMN, column], ascending=[True, False, True])
            .drop_duplicates(ADM2_COLUMN, keep="first")[[ADM2_COLUMN, column]]
        )
        table = table.merge(dominant, on=ADM2_COLUMN, how="left")
    table[OTTO_BASIN_COLUMNS] = table[OTTO_BASIN_COLUMNS].astype("string")
    logger.info("Built Otto basin assignments for %d ADM2 regions", len(table))
    return table


def write_otto_basin_table(gpkg_path, output_dir, layer=DEFAULT_RAW_GPKG_DRAINAGE_LAYER, trench_ids=None):
    """Build the Otto basin table and write it to `output_dir`.

    Without `trench_ids`, restricts to the trenches of an existing
    `river_network_trenches.parquet` in `output_dir` when there is one, so the
    table matches the processed network. When `output_dir` also holds the
    trench-to-ADM2 matches and drainage areas, the ADM2 assignment table is
    rebuilt alongside.
    """
    output_path = Path(output_dir)
    if trench_ids is None and (output_path / TRENCHES_FILENAME).exists():
        trench_ids = pd.read_parquet(output_path / TRENCHES_FILENAME, columns=[TRENCH_ID_COLUMN])[TRENCH_ID_COLUMN]
    table = build_otto_basin_table(gpkg_path, layer=layer, trench_ids=trench_ids)
    output_path.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path / OTTO_BASINS_FILENAME, index=False)
    logger.info("Wrote %s", output_path / OTTO_BASINS_FILENAME)

    trench_adm2_path = output_path / TRENCH_ADM2_TABLE_FILENAME
    drainage_areas_path = output_path / DRAINAGE_AREAS_FILENAME
    if trench_adm2_path.exists() and drainage_areas_path.exists():
        adm2_table = build_adm2_otto_basin_table(
            pd.read_parquet(trench_adm2_path),
            table,
            pd.read_parquet(drainage_areas_path, columns=[TRENCH_ID_COLUMN, DRAINAGE_AREA_COLUMN]),
        )
        adm2_table.to_parquet(output_path / ADM2_OTTO_BASINS_FILENAME, index=False)
        logger.info("Wrote %s", output_path / ADM2_OTTO_BASINS_FILENAME)
    return table
