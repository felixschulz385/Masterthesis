# Health stage: status

**Status (2026-10-21): parked.** Pollution is being finalized first. The descriptives
notebook runs end to end, but estimation is blocked by the data issues below. Resume by
working through the checklist, re-assembling `adm2_panel`, and re-running
`01_descriptives.ipynb`.

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

Ordered by priority. Each item names the code that needs to change.

| # | Issue | Evidence | Fix |
|---|---|---|---|
| 1 | Health outcomes are not in `adm2_panel` | §0; "Deferred" note at the end of `setup/assembly_datasets.yaml` | Add health sources to the `adm2_panel` config. Hospitalizations are long over `metric_name` (and ICD chapter), so they can use `type: long_pivot` with `filter: {metric_name: ...}`, `pivot_column: icd10_chapter_code`, and `id_map: {municipality_code: mun_id}`. Births are already wide. Then delete the temporary loaders in `src/analysis/panels.py` |
| 2 | Morbidity channels are (near) zero in every year from 2008 on | §0 series check: gastrointestinal about 572k in 2006 vs 7k in 2010; all other channels 0 | Fix the morbidity-list mapping for the current SIH system in `src/data/sources/health/preprocess/`. Until then, use ICD-10 chapters only |
| 3 | 217 municipalities with no upstream area are dropped from the panel | §5; the inner join on `LAND_COVER_COMPOSITION_BUCKET_MAP` in `src/data/sources/land_cover/composition.py` | Keep them with zero upstream exposure, plus an indicator. They include São Luís, São Gonçalo, Niterói and Florianópolis, and account for 4.4% of hospitalizations |
| 4 | Local (municipality-touching) land cover is dropped | §5: negative buckets are the only buckets with `bucket_intersects_adm2` set | Add a separate `local_lc_*` block from buckets < 0. Local and upstream changes are correlated (0.73–0.83 for forest, pasture and agriculture), so the separation has to be identified carefully |
| 5 | The ADM2 distance kernel is effectively flat | §5: Gaussian with `DEFAULT_ADM2_KERNEL_BANDWIDTH_KM = 1_000_000` gives weight 1.0 to every ring | Decide the bandwidth deliberately, either matching the sensor-side inv-sqrt kernel or setting a bandwidth in km (e.g. 100) |
| 6 | Some birth tables lose their modal bin in some years | §0: 37–41 weeks missing in 2003, 2006–07, 2009, 2012, 2016–18; 3000–3999 g missing in 2017 | Fix the bin-label harmonization in health preprocessing. The derived LBW and preterm shares are unaffected meanwhile |
| 7 | No mortality data | — | Fetch and preprocess the SIM `mortality` subtype |
| 8 | No basin id in `adm2_panel` for clustering | §7 | Attach each municipality's dominant Otto level-3 basin (by area) from `river_network_otto_basins.parquet` joined through `river_network_trench_adm2_matches.parquet` |
| 9 | Only upstream climate is available | §6 | Add local ERA5-Land climate per municipality as a direct confounder |

## Descriptive findings so far

- **Outcomes:**
  - ICD-chapter rates are complete and have few zeros, except congenital (14% zeros)
    and perinatal (6%).
  - Dispersion falls with population, which argues for population weights or PPML with
    population as exposure.
- **Identifying variation:** after municipality + state×year FE, about 0.4–2% of the
  variance in land-cover shares is left (ALR terms: 0.6–1.9%).
  - The long differences are substantial, though: 52% of municipalities saw upstream
    pasture shift by more than 5 percentage points between 2000 and 2023.
- **Inference:** 27 states, effective number about 14. Errors are likely correlated along
  rivers, which state clustering does not capture.
