"""Otto Pfafstetter sub-basin codes per trench.

ANA's BHO drainage-area layer carries each trench's Otto basin code at levels
1-6 (`nunivotto1` ... `nunivotto6`; level k is the first k digits of
`cobacia`). `river_network`'s connected-component `system_id` is too coarse
for clustering (one system holds ~42% of sensor samples), so these nested
basins are kept as a separate, attribute-only output rather than a column of
`river_network_trenches.parquet`: building it needs no geometry and no rerun of
the full network/matrix pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .constants import (
    DEFAULT_RAW_GPKG_DRAINAGE_LAYER,
    OTTO_BASINS_FILENAME,
    TRENCH_ID_COLUMN,
    TRENCHES_FILENAME,
)


logger = logging.getLogger(__name__)

OTTO_BASIN_LEVELS = (1, 2, 3, 4, 5, 6)


def otto_basin_column(level):
    """Output column name for the Otto basin code at `level`."""
    return f"otto_basin_l{level}"


OTTO_BASIN_COLUMNS = [otto_basin_column(level) for level in OTTO_BASIN_LEVELS]


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


def write_otto_basin_table(gpkg_path, output_dir, layer=DEFAULT_RAW_GPKG_DRAINAGE_LAYER, trench_ids=None):
    """Build the Otto basin table and write it to `output_dir`.

    Without `trench_ids`, restricts to the trenches of an existing
    `river_network_trenches.parquet` in `output_dir` when there is one, so the
    table matches the processed network.
    """
    output_path = Path(output_dir)
    if trench_ids is None and (output_path / TRENCHES_FILENAME).exists():
        trench_ids = pd.read_parquet(output_path / TRENCHES_FILENAME, columns=[TRENCH_ID_COLUMN])[TRENCH_ID_COLUMN]
    table = build_otto_basin_table(gpkg_path, layer=layer, trench_ids=trench_ids)
    output_path.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path / OTTO_BASINS_FILENAME, index=False)
    logger.info("Wrote %s", output_path / OTTO_BASINS_FILENAME)
    return table
