#!/usr/bin/env python3
"""Rebuild the SIH selected-morbidity-list manifest from the batch files on disk.

A fetch run whose current-SIH form loaded only morbidity-list values 1-2 planned
`water_sanitation_gastrointestinal` with 2 of its 6 values, dropped the other 14
channels for 2008+, and rewrote the manifest with that partial plan. The complete
batches of an earlier run (`__group_<n>` with n = the channel's full value count)
are still on disk but no longer listed, so preprocessing zero-filled the channels.

This script backs up the manifest and writes one entry per (source, year, channel):
`completed` when a full-group batch file exists, otherwise `pending`, so a resumed
`python -m src.data.sources.health fetch --subtype hospitalization` fetches only
the missing batches. Partial-group files are left on disk but not listed.

Usage: python scripts/repair_sih_morbidity_manifest.py [--root-dir .] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from datetime import datetime

from src.data.shared.batches import load_manifest, manifest_path, write_manifest
from src.data.sources.health.constants import HEALTH_DATASET_NAME
from src.data.sources.health.fetch.datasus import (
    SIH_ALL_METRICS_KEY,
    SIH_ICD10_YEAR_MIN,
    SIH_RESIDENCE_SOURCES,
    SIH_SELECTED_MORBIDITY_CHANNELS,
    SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID,
    _build_sih_manifest_entry,
)


def planned_years(entries):
    """Years to cover: the manifest's own years, from the morbidity-list start year."""
    return sorted({int(entry["export_year"]) for entry in entries if int(entry["export_year"]) >= SIH_ICD10_YEAR_MIN})


def source_key_for_year(year):
    for source_key, source in SIH_RESIDENCE_SOURCES.items():
        if source["year_min"] <= year and (source["year_max"] is None or year <= source["year_max"]):
            return source_key
    raise ValueError(f"No SIH source covers year {year}.")


def rebuild_entries(root_dir, years):
    entries = []
    for year in years:
        source_key = source_key_for_year(year)
        for channel, values in SIH_SELECTED_MORBIDITY_CHANNELS.items():
            entry = _build_sih_manifest_entry(
                root_dir,
                SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID,
                source_key,
                year,
                SIH_ALL_METRICS_KEY,
                morbidity_channel=channel,
                morbidity_filter_values=list(values),
                select_all_content=True,
            )
            if os.path.exists(entry["raw_path"]):
                entry["status"] = "completed"
            entries.append(entry)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    existing = load_manifest(args.root_dir, HEALTH_DATASET_NAME, SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID)
    if not existing:
        raise SystemExit("No existing manifest to repair.")
    entries = rebuild_entries(args.root_dir, planned_years(existing))

    print("existing:", dict(Counter(entry["status"] for entry in existing)))
    print("rebuilt: ", dict(Counter(entry["status"] for entry in entries)))
    for entry in entries:
        if entry["status"] != "completed":
            print(f"  pending: {entry['source_key']} {entry['export_year']} {entry['morbidity_channel']}")
    if args.dry_run:
        return

    path = manifest_path(args.root_dir, HEALTH_DATASET_NAME, SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID)
    backup_path = f"{path}.bak-{datetime.now():%Y%m%d-%H%M%S}"
    shutil.copy2(path, backup_path)
    write_manifest(args.root_dir, HEALTH_DATASET_NAME, SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID, entries)
    print(f"Backed up {path} to {backup_path} and wrote {len(entries)} entries.")


if __name__ == "__main__":
    main()
