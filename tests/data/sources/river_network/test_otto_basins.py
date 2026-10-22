from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.data.sources.river_network.__main__ import configure_parser, run
from src.data.sources.river_network.constants import (
    ADM2_OTTO_BASINS_FILENAME,
    DRAINAGE_AREAS_FILENAME,
    OTTO_BASINS_FILENAME,
    TRENCH_ADM2_TABLE_FILENAME,
    TRENCHES_FILENAME,
)
from src.data.sources.river_network.otto_basins import (
    OTTO_BASIN_COLUMNS,
    build_adm2_otto_basin_table,
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


def _otto_codes(codes_by_trench):
    return pd.DataFrame(
        {
            "trench_id": list(codes_by_trench),
            **{
                f"otto_basin_l{level}": [code[:level] for code in codes_by_trench.values()]
                for level in range(1, 7)
            },
        }
    )


def test_build_adm2_otto_basin_table_picks_largest_drainage_area_per_level() -> None:
    # ADM2 "A" touches trenches in basins 745 (area 5) and 714 (areas 3 + 4):
    # 714 wins at level 3 on summed area although 745 holds the largest trench.
    otto = _otto_codes({1: "745539", 2: "714925", 3: "714111", 4: "891282"})
    trench_adm2 = pd.DataFrame({"trench_id": [1, 2, 3, 4], "adm2": ["A", "A", "A", "B"]})
    drainage = pd.DataFrame({"trench_id": [1, 2, 3, 4], "drainage_area": [5.0, 3.0, 4.0, 1.0]})

    table = build_adm2_otto_basin_table(trench_adm2, otto, drainage).set_index("adm2")

    assert table.loc["A", "otto_basin_l3"] == "714"
    assert table.loc["A", "otto_basin_l6"] == "745539"
    assert table.loc["B", "otto_basin_l3"] == "891"
    assert pd.api.types.is_string_dtype(table["otto_basin_l3"])


def test_write_otto_basin_table_writes_adm2_table_when_matches_exist(tmp_path: Path) -> None:
    gpkg_path = _drainage_gpkg(tmp_path)
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    pd.DataFrame({"trench_id": [1, 3], "adm2": ["A", "A"]}).to_parquet(output_dir / TRENCH_ADM2_TABLE_FILENAME)
    pd.DataFrame({"trench_id": [1, 3], "drainage_area": [1.0, 2.0]}).to_parquet(output_dir / DRAINAGE_AREAS_FILENAME)

    write_otto_basin_table(gpkg_path, output_dir, layer=DRAINAGE_LAYER)

    written = pd.read_parquet(output_dir / ADM2_OTTO_BASINS_FILENAME)
    assert written["adm2"].tolist() == ["A"]
    assert written["otto_basin_l3"].tolist() == ["891"]


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
