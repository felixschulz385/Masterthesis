# Health stage: status

**Status (2026-10-21): parked.** Pollution is being finalized first. Health outcomes and
Otto basins are now in `adm2_panel` (blockers 1 and 8), the ADM2 kernel matches the sensor side (5), and local land cover, local climate and headwater municipalities are in the panel (3, 4, 9), and the descriptives
notebook runs end to end on it. Estimation is still blocked by the remaining data issues
below. Resume by working through the checklist, re-assembling `adm2_panel`, and
re-running `01_descriptives.ipynb`.

## Design

- **Question:** does upstream land-cover change affect health in downstream municipalities?
- **Unit:** municipality (6-digit IBGE `mun_id`) × year, from `data/assembly/adm2_panel.parquet`.
- **Window:** 2000–2023. Population (the denominator) starts in 2000; SINASC ends in 2023.
- **Candidate outcomes:**
  - SIH hospitalization rates per 1,000 by ICD-10 chapter: 01 infectious/parasitic,
    11 digestive, 14 genitourinary, 16 perinatal, 17 congenital; 19 injury as placebo.
  - Low-birth-weight share and preterm share (SINASC).
  - All-cause hospitalization and in-hospital death rates.
- **Exposure:** land cover in 25 km rings upstream of the municipality (`lc_*` / `alr_*`).

## Blockers

Ordered by priority. All three need a DATASUS re-download, which runs on a local machine
(scicore has no internet access): see [`health_refetch_handoff.md`](../../../health_refetch_handoff.md).

| # | Issue | Evidence | Fix |
|---|---|---|---|
| 2 | Morbidity channels are missing for most channels in 2021–2024, plus pregnancy 2009 and renal 2013 (48 batches) | `scripts/repair_sih_morbidity_manifest.py --dry-run` lists them as `pending` | Resume the fetch locally: `python -m src.data.sources.health fetch --subtype hospitalization` plans only the pending batches |
| 6 | Some birth tables lose their modal bin in some years | §0: 37–41 weeks missing in 2003, 2006–07, 2009, 2012, 2016–18; 3000–3999 g missing in 2017 | Re-download SINASC (`--subtype birth`; the raw files are not on scicore), then fix the bin-label harmonization in `_clean_birth_outcome_frame`. The derived LBW and preterm shares are unaffected meanwhile |
| 7 | No mortality data | — | Download SIM deaths by municipality × age band (`--subtype mortality`), then add them to `adm2_panel` (`under_1` gives infant deaths) |

Resolved:
- **1. Health outcomes in `adm2_panel`.** `adm2_panel` now carries SIH hospitalizations
  (`hosp_*_count` all-cause, `hosp_icd_01` … `hosp_icd_99` by ICD-10 chapter, and
  `hosp_<channel>` morbidity channels, via `long_pivot` with `rename`/`column_template`)
  and SINASC births (`births`, `births_low_weight`, `births_preterm` plus their `*_known`
  denominators, from `health_births.parquet`, built by
  `python -m src.data.sources.health preprocess --subtype birth_summary`). The values
  match the former temporary loaders exactly. `panels.hospitalization_columns()` and
  `panels.birth_columns()` name the column groups.
- **5. Distance kernel.** Both ADM2 sources (land cover, climate) use the sensor side's
  inv-sqrt kernel, which is also `DEFAULT_ADM2_DISTANCE_KERNEL` now. The former Gaussian
  with bandwidth 1,000,000 km weighted every ring equally; within-municipality
  land-cover changes under the two kernels correlate at 0.986 or more. The pollution
  arm's 200 km reach cap is applied in analysis code and is still a health model-setup
  decision.
- **2. Morbidity channels after 2008 (mostly).** They were (near) zero from 2008 on
  because a fetch run whose current-SIH form loaded only morbidity-list values 1–2
  planned gastrointestinal with 2 of its 6 values, silently dropped the other 14
  channels, and rewrote the manifest; preprocessing then zero-filled the unlisted
  batches. The complete batches of an earlier run were still on disk.
  `scripts/repair_sih_morbidity_manifest.py` rebuilt the manifest from them (backup
  `manifest.jsonl.bak-20261021-162620`); the fetch now refuses to plan partial channels,
  and `_complete_panel` zero-fills only within fetched batches, so unfetched batches stay
  missing (this also turned ICD chapter 99 into NaN in years where it does not exist, and
  filled legacy 1999 injuries and 2001 toxic poisoning, which were zero). Channels are now
  continuous across 2007/2008 (2010/2006 ratios 0.87–1.35, except vector-borne at 2.3,
  the 2010 dengue epidemic). Caveat: legacy-2000 injuries lists fewer causes than other
  years.
- **3, 4, 9. Local exposure and headwater municipalities.** Bucketed ADM2 sources take
  `scope: local`, which averages (equal weight) the rings of the trenches touching the
  municipality (`bucket_intersects_adm2`) into `local_lc_*`, `local_alr_*` and
  `local_<climate>` columns. The local land-cover source is the panel's spine, so the
  217 headwater municipalities (3.8% of hospitalizations) stay in with NaN upstream
  exposure; `panels.load_adm2_panel()` adds `has_upstream`. Local and upstream land-cover
  changes 2000–2023 correlate at 0.78–0.86 for forest, pasture and agriculture, and the
  within-FE ALR terms at 0.19–0.51. Local and upstream climate correlate at 0.79–0.94
  within municipality. Two coastal municipalities have no local climate.
- **8. Basin id for clustering.** `adm2_panel` carries `otto_basin_l2` … `otto_basin_l5`,
  the municipality's dominant basin by summed drainage area of its intersecting trenches
  (`river_network_adm2_otto_basins.parquet`, written by `python -m
  src.data.sources.river_network otto-basins`). Otto 3: 217 basins over the panel's
  municipalities, effective about 39.

## Descriptive findings so far

- **Outcomes:**
  - ICD-chapter rates are complete and have few zeros, except congenital (14% zeros)
    and perinatal (6%).
  - Dispersion falls with population, which argues for population weights or PPML with
    population as exposure.
- **Identifying variation:** after municipality + state×year FE, about 0.4–2% of the
  variance in land-cover shares is left (ALR terms: 0.6–1.9%).
  - The long differences are substantial, though: 53% of municipalities saw upstream
    pasture shift by more than 5 percentage points between 2000 and 2023.
- **Inference:** 27 states, effective number about 14. Errors are likely correlated along
  rivers, which state clustering does not capture; Otto-3 basins (217, effective about 39)
  do.
