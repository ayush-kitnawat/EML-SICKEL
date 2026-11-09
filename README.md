# SICKLE EML Project: Temporal Importance for Paddy Yield Estimation

## Latest reproducible findings (9 November 2026)

The cutoff-safe 15–90-day experiments, within-plot differences, matched
2019/2020 transfer, and **one frozen, reserved day-45 comparison** are complete.
See [roadmap execution and safeguards](ROADMAP_RUN.md) for the research protocol,
limitations, and full aggregate results. Machine-readable, non-identifying
scores are saved in [aggregate metrics](results/roadmap_aggregate_metrics.csv)
and [paired validation differences](results/validation_paired_differences.csv).

On the 37-row / 14-plot reserved validation cohort, agronomic Ridge achieved
533.23 RMSE (kg/acre), PCA + Ridge with S1/S2 achieved 548.13, and the
controls-only Extra Trees benchmark achieved 505.30. **Satellite inputs did not
improve the frozen EML prediction benchmark.** Prior work had already examined
the official validation set for Ridge, so it is not a pristine independent
external validation cohort. Raw data, row-level labels/predictions, and
generated outputs are deliberately not tracked by Git.

## Research question

Which **observation periods** and **sensor-stage combinations** are most predictive of
final farmer-reported paddy yield? How early can final yield be predicted without
substantially degrading out-of-plot performance?

## Local data layout

- `data/archives/`: preserved downloaded ZIPs (full, toy, pretrained weights).
- `data/raw/sickle_full/`: complete SICKLE dataset after extraction.
- `data/raw/sickle_toy/`: small SICKLE dataset for prototype work.
- `data/weights/runs/`: SICKLE authors' pretrained deep learning baselines.
- `scripts/audit_dataset.py`: standard-library metadata/yield-label audit.
- `scripts/extract_features.py`: one masked, quality-filtered row per satellite scene.
- `scripts/build_stage_features.py`: stage-wise plot-season regression data.
- `scripts/stage_importance_baseline.py`: nested plot-grouped Ridge stage ablations.
- `scripts/compare_regressors.py`: grouped OLS/Ridge/LASSO/Elastic Net comparison.
- `scripts/selection_stability.py`: repeated plot-grouped LASSO selection study.
- `scripts/extended_experiments.py`: sensors, mask geometry, Landsat pixel-size,
  10 narrower windows, cloud thresholds and **prospective day cutoffs**.
- `scripts/econometric_extensions.py`: test incremental satellite information
  beyond known year, area, variety and season.
- `scripts/early_vs_survey_controls.py`: prospective satellite observations
  versus a practical agronomic-information baseline.
- `scripts/coverage_and_contrasts.py`: coverage, plot size and paired uncertainty.
- `scripts/grouped_learning_curve.py`: vary the number of labelled training plots.
- `outputs/eda/`: generated observation metadata and audit summaries.
- `outputs/features/`: per-scene features and stage-wise feature matrices.
- `outputs/models/`: model predictions, metrics, ablation tables, and figures.
- `outputs/robustness/`: extended experiment predictions, analyses and figures.
- `EXPERIMENT_RESULTS.md`: consolidated, source-audited preliminary findings.

**Do not commit** `data/` or large generated outputs to version control.
The supplied `weights.zip` is for reproducing the original SICKLE deep learning
benchmarks; OLS/Ridge/LASSO/Elastic Net **do not require pretrained weights**.

## First audit (no third-party Python packages needed)

```bash
cd ~/Desktop/College/sem-7/EML/EML-SICKEL
python3 scripts/audit_dataset.py --dataset toy
python3 scripts/audit_dataset.py --dataset full
```

The script reads the released `sickle_dataset_tabular.csv`, keeps paddy samples
with positive reported yield and valid sowing/harvesting dates, and scans the
corresponding NPZ acquisition **filenames**, not the millions of individual
TIFF band files. Results include label counts, plot split overlap, stage/sensor
coverage, and a plot-season/image-date CSV in `outputs/eda/`.

### Verified first audit (25 October 2026)

| Quantity | Full dataset | Toy dataset |
|---|---:|---:|
| All records | 2,370 | 217 |
| Paddy records | 907 | 89 |
| Positive-yield paddy records with valid sow/harvest dates | **319** | **34** |
| Physical plots in this usable subset | **131** | **29** |
| Train/val positive-yield rows | 282 / 37 | 28 / 6 |
| Train/val physical plot overlap | 0 | 0 |

For the **319 usable full-data plot-seasons**, in-season NPZ observation counts:
S1 **4,899**, S2 **14,293**, L8 **3,536**. These are *acquisition availability*
counts, not yet cloud- or pixel-quality-filtered observations. Stage coverage
details are in `outputs/eda/full_summary.json`.

The official paper describes 351 richly annotated paddy records, but the
downloaded CSV yields 319 records meeting the explicit positive-yield and date
filters. Do not silently use 351 as the regression sample size.

## Run the implemented feature extraction and baseline

The existing Conda environment `~/miniconda3/envs/global` is used to run the
pipeline. `rasterio` is installed in that environment; other required libraries
are listed in `requirements.txt`.

```bash
cd ~/Desktop/College/sem-7/EML/EML-SICKEL
PY=~/miniconda3/envs/global/bin/python

# Verify algorithms and built outputs (no model training)
$PY -m unittest discover -s tests -v

# Data audit; already completed on full and toy data
$PY scripts/audit_dataset.py --dataset toy
$PY scripts/audit_dataset.py --dataset full

# Feature extraction (writes all observation records, including failures)
$PY scripts/extract_features.py --dataset toy
$PY scripts/extract_features.py --dataset full

# Aggregate five 20%-duration windows into 55 spectral/radar features
$PY scripts/build_stage_features.py --dataset toy
$PY scripts/build_stage_features.py --dataset full

# Train-only grouped CV with nested Ridge tuning; stage ablation & figures
$PY scripts/stage_importance_baseline.py --dataset full

# Train-only grouped CV comparison; OLS, Ridge, LASSO, Elastic Net
$PY scripts/compare_regressors.py

# 40 repeated plot-grouped 80%-subsample LASSO fits; stability heatmap
$PY scripts/selection_stability.py --repeats 40

# Run cloud sensitivity: ~14k S2 scenes are reprocessed per threshold.
# Alternative output tags prevent overwriting the canonical feature CSV.
$PY scripts/extract_features.py --dataset full --sensors S2 --s2-cloud-threshold 35 --tag s2cloud35
$PY scripts/extract_features.py --dataset full --sensors S2 --s2-cloud-threshold 90 --tag s2cloud90

# Extended research suite: 42 grouped-CV model scenarios, figures and predictions
$PY scripts/extended_experiments.py
$PY scripts/econometric_extensions.py
$PY scripts/early_vs_survey_controls.py
$PY scripts/coverage_and_contrasts.py
$PY scripts/grouped_learning_curve.py --repeats 4

# All 11 unit and data integrity tests
$PY -m unittest discover -s tests -v
```

### Verified feature extraction (full dataset, 25 October 2026)

| Sensor | Within-season scene files | Accepted quality-filtered plot scenes |
|---|---:|---:|
| Sentinel-1 | 4,899 | 4,646 |
| Sentinel-2 | 14,293 | 6,447 |
| Landsat-8 | 3,536 | 1,159 |
| **Total** | **22,728** | **12,252** |

The 319 labelled plot-seasons produce 55 spectral/radar stage predictors. The
feature-extraction summary (`outputs/features/full_observation_features.summary.json`)
also reports rejected observations. Valid observations are **not** the same as
ground-truth yield prediction performance.

The masks' first of six layers encodes `PLOT_ID` (confirmed against official
`reference/sickle_official/utils/dataset.py`). Sentinel-1/2 use the 10 m mask,
Landsat-8 the 30 m mask. S2 SCL class 4/5/6/7 and probability <=65 are allowed.
Landsat QA rejects fill/cloud/shadow/snow and saturated red/NIR bands. L8 SR
pixels are converted using **DN × 0.0000275 − 0.2** before NDVI calculation.
Sentinel-1 VV/VH values are used in dB; S2 indices use the released bands.

**Limitations to resolve before claiming final scientific results:** 1–2 px
NPZ/plot-mask size differences are nearest-neighbor resampled (matching the
official code's practical resizing idea), but original NPZ georeferencing is
not provided for independent pixel-perfect verification. Inspection of the
released Sentinel-2 TIFFs showed **placeholder/local pixel-coordinate transforms**
that are inadequate for reconstructing their true geographical relative offset.
The released NPZ arrays agree pixel-for-pixel with inspected corresponding TIFF
bands where dimensions match, but the metadata do not establish accurate
cross-scene or cross-tile alignment. Scenes with larger
spatial shape mismatches are rejected. Small Landsat plots can have no 30 m
pixels; do not borrow neighboring-plot values. The current SCL/cloud-threshold
settings are defensible first-pass QC choices, not author-validated optimums.

### Initial, corrected econometric baseline (full dataset)

Training CV: **282 rows from 117 distinct plots**, five grouped outer folds,
three inner grouped folds for hyperparameter search. The **37-row / 14-plot**
official validation split is never used to select model features, stages or
hyperparameters. Only the **exact 55 spectral/radar measurements** are passed
to the models (NOT scene counts, quality fractions, or season dates).

| Model | Grouped CV RMSE | Grouped CV MAE | Grouped CV R² |
|---|---:|---:|---:|
| OLS | 415.2 | 325.7 | −0.080 |
| Ridge | **387.5** | 290.8 | **0.059** |
| LASSO | 394.3 | 294.9 | 0.026 |
| Elastic Net | 388.1 | **289.0** | 0.057 |

OLS's poorer score is expected when predictors are highly correlated with
limited labels; it is not evidence any particular stage is biologically unimportant.
This is a **weak** predictive signal, not a high-accuracy yield system.
The full-stage Ridge model's held-out official VAL scores are in
`outputs/models/full_ridge_summary.json` (RMSE ~579.7; R² ~0.147).

Stage-only Ridge CV suggests **W5 (80–100%)** and **W1 (0–20%)** are the
strongest standalone windows (RMSE ~393.2 and ~396.2 respectively; full ~387.5).
Removing W1 increases OOF RMSE by ~12.6; removing W5 by ~8.3. Removing W2 or W3
*improves* this specific Ridge CV score. The paired plot-bootstrap intervals
are uncertain and are conditional on already-fitted OOF predictions; they do
not establish stage-specific statistical or causal significance.

Across 40 plot-grouped 80%-subsample LASSO fits, average feature selection
frequency was highest for W1 (~0.398) and W5 (~0.366), followed by W4 (~0.284),
W2 (~0.198), W3 (~0.148). Collinear features can substitute for one another;
selection frequency is not proof of a biological mechanism.

These are **retrospective normalized-season** results: known harvest dates were
used only to define the window positions. A real early-season forecast would
require window cutoffs defined without the future realized harvest date.
The paired plot-bootstrap intervals provided alongside ablation results are
conditional on frozen OOF predictions, not full refit-based confidence bounds.

### Feature extraction prerequisites / cautions

- Use `.npz` satellite images, not millions of separate `.tif` band files.
- `masks/10m/*.tif` and `masks/30m/*.tif` are multi-band TIFFs; use
  `rasterio` to inspect/select the actual plot mask. Pillow/OpenCV do not
  reliably load all their channels.
- Verify the SICKLE code's mask-band semantics and pixel grids *before*
  applying masks to NPZ arrays (even images from the same sensor can differ
  slightly in shape).
- Use Sentinel-2 `SCL`/cloud-probability and Landsat `QA_PIXEL` for quality
  filtering, with correctly matched resolutions.
- Verify Landsat Level-2 surface-reflectance **scale and offset** before NDVI;
  raw SR digital numbers must not be treated as physical reflectance.
- Pretrained weights under `data/weights/runs/` are optional SICKLE neural
  network baselines, not required for regularized econometric regression.

## Planned research workflow

1. Validate the dataset audit and yield units, plot masks, band values,
   pixel scaling, and optical cloud/quality flags against the SICKLE code.
2. For each eligible plot-season and sensor, extract masked plot-level
   vegetation/backscatter summaries from `.npz` images. Preserve the date and
   valid-pixel count for every observation; avoid pixel-level data leakage.
3. Partition *retrospective* growing seasons into five normalized time windows.
   Check image coverage before selecting features/aggregation.
4. Aggregate observation-level features by window (e.g., median, slope,
   variability, acquisition count), with missingness indicators.
5. Compare OLS, Ridge, LASSO, and Elastic Net with preprocessing and tuning
   **inside grouped folds**; group by `PLOT_ID` and check original split overlap.
6. Run stage-only, leave-one-stage-out, and cumulative-information experiments,
   holding outer validation splits fixed. Report MAE/RMSE/R² and uncertainty.
7. Assess stability under grouped resampling; visualize importance by period
   and by sensor. Do not claim causal effects from predictive feature rankings.
8. For true *early* forecasts, use only observations and time features available
   as of the forecast date; do **not** normalize using the future realized
   harvesting date (that is fine only for retrospective stage comparisons).

See the accompanying project proposal for the original econometric framing.

## Extended EML experiment results

See **`EXPERIMENT_RESULTS.md`** and `outputs/robustness/` for measured results,
paired contrasts, early forecasting, 10-window ablations, area-stratified
observation coverage, satellite-versus-survey controls and the grouped learning
curve. The nominal bootstrap bounds in these files are conditional on fixed
out-of-fold model predictions; they should not be presented as strict inferential
tests after multiple model/window comparisons.
