# Analysis notebooks

Upstream land-cover change → river pollution → health, estimated in two linked arms:

| Arm | Question | Panel (index) | Exposure |
|---|---|---|---|
| `pollution/` | Does upstream land-cover change degrade water quality? | `data/assembly/sensor_panel.parquet` (`station_code`, `datetime`) | land cover in 25 km rings upstream of the station |
| `health/` | Does upstream land-cover change affect health downstream? | `data/assembly/adm2_panel.parquet` (`mun_id`, `year`) | land cover in 25 km rings upstream of the municipality |

## Workflow

Each arm has the same two steps:

1. **`01_descriptives.ipynb`**: the descriptive evidence needed to *choose* the model
   (sample, outcomes, transforms, FE, exposure definition, kernel, controls, clustering).
   Every section ends with an **→ Informs** note, and the final section is a decision
   table to fill in.
2. **`02_analysis.ipynb`**: estimation, implementing the decisions from step 1.
   `pollution/02_analysis` is built on `src/analysis/pollution.py`; `health/02_analysis`
   is a stub.

Re-run a descriptives notebook after any re-assembly, and record changed decisions in
its table before touching `02_analysis`.

Other notebooks:
- `health/upstream_mining_municipality_map.ipynb`: municipality upstream-deforestation
  map (moved from `src/analysis/notebooks/`; its `land_cover.feather` input is stale).

## Shared code

- `src/analysis/panels.py`: panel loaders (restricted to the 1985–2024 exposure window),
  variable groups derived from `setup/assembly_datasets.yaml`, and temporary health
  loaders (see below).
- `src/analysis/pollution.py`: the pollution model setup (sample rules, outcome cleaning,
  combined fecal indicator, CONAMA 357 exceedances, land cover recomputed within 200 km
  and per distance band). `build_pollution_panel()` returns the estimation frame.
- `src/analysis/inference.py`: wild-cluster bootstrap for high-dimensional FE models.
- `src/analysis/descriptives.py`: FE demeaning (pyfixest), variation tables, distribution
  tables, binned scatters, effective cluster counts.

The maintained CLI regression pipeline (`python -m src.cli analysis sensor-data`) lives
in `src/analysis/sensor_data/` and is separate from these notebooks.

## Status (as of 2026-10-21)

**Pollution: active.** The model setup is decided (§11 of `pollution/01_descriptives.ipynb`)
and implemented in `pollution/02_analysis.ipynb`, with standard errors clustered on Otto
level-3 sub-basins (below). Coefficients are exported to
`output/analysis/pollution/coefficients.csv`.

**Health: parked.** Estimation is blocked by data issues: health outcomes are not in
assembly, morbidity channels are broken from 2008 on, and the exposure definition drops
headwater municipalities and local land use. See [`health/README.md`](health/README.md)
for the blockers and the resume checklist.

## Sub-basins (Otto Pfafstetter)

`river_network_otto_basins.parquet` (in `data/river_network/processed/`) maps every
`trench_id` to its Otto basin code at levels 1–6 (`otto_basin_l1` … `otto_basin_l6`,
string codes). The codes come from the attributes of the BHO drainage-area layer, so the
table needs no geometry and no rerun of the network/matrix pipeline:

```bash
python -m src.data.sources.river_network otto-basins   # ~10 s; also written by a full preprocess
```

`sensor_panel` carries `otto_basin_l2` … `otto_basin_l5` via the `otto_basins` assembly
source, and the pollution analysis clusters on `otto_basin_l3`. Cluster counts over
sensor samples:

| Level | Basins | Effective (inverse HHI) | Largest share |
|---|---|---|---|
| `system_id` | 261 | 5.0 | 42% |
| Otto 2 | 42 | 5.3 | 33% |
| **Otto 3** (used) | 207 | 27.2 | 9% |
| Otto 4 | 843 | 98.2 | 4% |
| Otto 5 | 2,276 | 347.2 | 2% |

Because some outcome samples have only a few dozen basins, §5.7 of
`pollution/02_analysis.ipynb` adds restricted wild-cluster bootstrap p-values (9,999
reps, Otto 3) for every land-cover coefficient (`src/analysis/inference.py`). pyfixest's
built-in `wildboottest` is infeasible here: it builds a dense one-hot matrix of all
~5,700 station FE. The module therefore bootstraps the FE-demeaned design, which is
equivalent by Frisch–Waugh–Lovell. Results are in
`output/analysis/pollution/wild_cluster_bootstrap.csv`.

Still open:
- **Health:** assign each municipality its dominant Otto-3 basin (by area) in the ADM2
  assembly (`health/README.md`, blocker 8).

## Mining arm

Mining is not in the baseline: 56% of station-years have no mining upstream, and only
2.6% of stations switch. It is estimated separately in §6 of
`pollution/02_analysis.ipynb`:
- **Sample:** the 1,205 stations with any mining within 50 km upstream (158 switch
  presence).
- **Regressors:** mining presence (`mining_present_50`) and the mined share of the
  50 km reach (`mining_pp_50`), on top of the baseline specification.
- **Outcomes:** turbidity, TSS, conductivity, pH, DO and the CONAMA metal exceedances.
- **Inference:** wild-cluster bootstrap, since the metal samples have only 12–20
  Otto-3 basins.

**Headline result:** mining appearing upstream raises mercury violations by about
14 pp (bootstrap p = 0.004). Adding 200 km `alr_mining` to the full-sample baseline
moves the other land-cover coefficients by at most 0.33 SE, so leaving mining out of the
baseline does not bias them.
