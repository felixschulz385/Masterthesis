from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.data.sources.river_network.__main__ import configure_parser, run
from src.data.sources.river_network.constants import OTTO_BASINS_FILENAME, TRENCHES_FILENAME
from src.data.sources.river_network.otto_basins import (
    OTTO_BASIN_COLUMNS,
    build_otto_basin_table,
    write_otto_basin_table,
)

DRAINAGE_LAYER = "drainage"


def _drainage_gpkg(tmp_path: Path) -> Path:
    """Tiny BHO-like drainage layer: trench 2 is duplicated, as in the real file."""
    codes = ["745539", "714925", "714925", "891282"]
    frame = gpd.GeoDataFrame(
        {
            "cotrecho": [1, 2, 2, 3],
            **{f"nunivotto{level}": [code[:level] for code in codes] for level in range(1, 7)},
            "geometry": [Point(i, i) for i in range(4)],
        },
        crs=4326,
    )
    path = tmp_path / "bho.gpkg"
    frame.to_file(path, layer=DRAINAGE_LAYER, driver="GPKG")
    return path


def test_build_otto_basin_table_renames_dedups_and_keeps_string_codes(tmp_path: Path) -> None:
    table = build_otto_basin_table(_drainage_gpkg(tmp_path), layer=DRAINAGE_LAYER)

    assert list(table.columns) == ["trench_id", *OTTO_BASIN_COLUMNS]
    assert table["trench_id"].tolist() == [1, 2, 3]
    # Codes must stay strings: numeric parsing would merge e.g. "07" with "7".
    assert table.loc[table["trench_id"] == 1, "otto_basin_l3"].item() == "745"
    assert pd.api.types.is_string_dtype(table["otto_basin_l6"])


def test_write_otto_basin_table_restricts_to_processed_trenches(tmp_path: Path) -> None:
    gpkg_path = _drainage_gpkg(tmp_path)
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    pd.DataFrame({"trench_id": [1, 3]}).to_parquet(output_dir / TRENCHES_FILENAME)

    write_otto_basin_table(gpkg_path, output_dir, layer=DRAINAGE_LAYER)

    written = pd.read_parquet(output_dir / OTTO_BASINS_FILENAME)
    assert written["trench_id"].tolist() == [1, 3]


def test_otto_basins_action_skips_the_full_network_pipeline() -> None:
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    args = parser.parse_args(["otto-basins", "--gpkg-path", "bho.gpkg", "--output-dir", "out"])

    with patch("src.data.sources.river_network.__main__.write_otto_basin_table") as write_table, patch(
        "src.data.sources.river_network.__main__.RiverNetwork"
    ) as network_cls:
        run(args)

    write_table.assert_called_once_with("bho.gpkg", "out")
    network_cls.assert_not_called()
