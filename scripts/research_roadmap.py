#!/usr/bin/env python3
"""EML-SICKEL research phases 1–5: audit, cutoff-safe EML, panel, and freeze.

Drop this file into the repository's scripts/ directory. Uses the existing
outputs/features/full_stage_features.csv and full_observation_features.csv.
No official VAL outcomes are used by audit/cutoffs/panel/transfer/freeze. The explicit
'validate' command reads VAL once after a frozen, SHA256-locked manifest.

The satellite feature construction for cutoff day d uses only acquisitions
whose image_date is between sowing_date and sowing_date+d (inclusive), and
never uses the realized harvest date or relative_season_time in predictors.
The observed harvest date is used *only* to exclude plots already harvested
at the cutoff from the common, comparable cohort.

These exploratory analyses were conceived after prior work on TRAIN; no CV
result here should be represented as unbiased independent model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "roadmap"
STAGE_FILE = ROOT / "outputs" / "features" / "full_stage_features.csv"
OBS_FILE = ROOT / "outputs" / "features" / "full_observation_features.csv"
CUTS = (15, 30, 45, 60, 75, 90)
PRIMARY_DAY = 45
PAIR_SOWING_MAX_CIRCULAR_DAYS = 45
S1 = ("s1_vv_db", "s1_vh_db", "s1_vv_minus_vh_db")
S2 = ("s2_ndvi", "s2_ndmi", "s2_ndre", "s2_red_reflectance", "s2_nir_reflectance")
SATELLITE_FEATURES = S1 + S2  # predeclared, 8 raw inputs x median/latest
SAT_COLUMNS = tuple(f"{f}__{stat}" for f in SATELLITE_FEATURES
                    for stat in ("median", "latest"))
C_NUM = ("area", "year")
C_CAT = ("season", "variety")
CORE_MODELS = ("ridge_controls", "ridge_s1s2", "pca_ridge_s1s2",
               "partial_elastic_s1s2", "extra_trees_controls")
FINAL_MODELS = ("ridge_controls", "pca_ridge_s1s2",
                "partial_elastic_s1s2", "extra_trees_controls")
SOURCE_NOTE = (
    "Sani et al. (WACV 2024), SICKLE supplementary, Fig. 4d: yield (kg/acre); "
    "https://openaccess.thecvf.com/content/WACV2024/supplemental/"
    "Sani_SICKLE_A_Multi-Sensor_WACV_2024_supplemental.pdf"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def load_frame(path: Path, columns=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required local data missing: {path}")
    return pd.read_csv(path, dtype={"unique_id": str, "plot_id": str},
                       usecols=columns, low_memory=False)


def load_stage(train_only=True) -> pd.DataFrame:
    frame = load_frame(STAGE_FILE)
    assert frame.unique_id.is_unique, "Duplicate seasonal IDs"
    required = set((*C_NUM, *C_CAT, "unique_id", "plot_id", "split",
                    "yield", "sowing_date", "harvesting_date"))
    assert required.issubset(frame.columns), required - set(frame.columns)
    assert set(frame.split.dropna()).issubset({"train", "val"})
    train = frame.loc[frame.split.eq("train")]
    val = frame.loc[frame.split.eq("val")]
    assert set(train.plot_id).isdisjoint(val.plot_id)
    assert train.plot_id.notna().all()
    # Original project reproducibility anchors: fail loudly if input changes.
    assert len(train) == 282 and train.plot_id.nunique() == 117
    assert len(val) == 37 and val.plot_id.nunique() == 14
    if train_only:
        return train.reset_index(drop=True)
    return frame


def load_obs() -> pd.DataFrame:
    # Keep complete metadata for detailed post-cutoff leakage audit.
    frame = load_frame(OBS_FILE)
    required = {"unique_id", "sensor", "status", "image_date",
                *SATELLITE_FEATURES}
    assert required.issubset(frame.columns), required - set(frame.columns)
    return frame


def verify_observation_times(observations: pd.DataFrame,
                             labels: pd.DataFrame) -> dict:
    """Crosscheck acquisition timestamps against true dates, not model labels."""
    if not labels.unique_id.is_unique:
        raise ValueError("Stage rows require unique_id")
    info = labels[["unique_id", "sowing_date"]].copy()
    combined = observations.merge(info, on="unique_id", how="inner",
                                  validate="many_to_one")
    assert len(combined) == len(observations.loc[
        observations.unique_id.isin(labels.unique_id)])
    image = pd.to_datetime(combined.image_date, errors="coerce").dt.normalize()
    sow = pd.to_datetime(combined.sowing_date, errors="coerce").dt.normalize()
    if image.isna().any() or sow.isna().any():
        raise ValueError("Malformed acquisition/sowing date")
    offsets = (image - sow).dt.days
    if "days_since_sowing" in combined:
        prior = pd.to_numeric(combined.days_since_sowing, errors="coerce")
        # Strong invariant. If not true, the upstream script may be misaligned.
        if prior.notna().any() and not (prior[prior.notna()].astype(int) ==
                                       offsets[prior.notna()]).all():
            raise ValueError("Stored days_since_sowing disagrees with image_date")
    return {"n_observation_rows": int(len(combined)),
            "pre_sowing_acquisitions": int((offsets < 0).sum()),
            "post_90_day_acquisitions": int((offsets > 90).sum()),
            "minimum_image_offset": int(offsets.min()) if len(offsets) else None,
            "maximum_image_offset": int(offsets.max()) if len(offsets) else None}


def cutoff_features(observations: pd.DataFrame, labels: pd.DataFrame,
                    cutoff: int, sensors: tuple[str, ...] = ("S1", "S2")) -> pd.DataFrame:
    """Build strictly prospective, date-scoped median and most recent values.

    Important: both median and latest use real acquisition image dates, and
    sorting/ties do not use relative_season_time, harvest date, or outcomes.
    """
    if cutoff < 0:
        raise ValueError("Negative forecasting cutoff")
    if not labels.unique_id.is_unique:
        raise ValueError("unique_id must be unique in the label cohort")
    info = labels[["unique_id", "sowing_date"]]
    raw = observations.merge(info, on="unique_id", how="inner",
                             validate="many_to_one")
    image = pd.to_datetime(raw.image_date, errors="coerce").dt.normalize()
    sow = pd.to_datetime(raw.sowing_date, errors="coerce").dt.normalize()
    if image.isna().any() or sow.isna().any():
        raise ValueError("Acquisition or sow date cannot be parsed")
    offset = (image - sow).dt.days
    accepted = raw.loc[(raw.status.eq("ok")) & raw.sensor.isin(sensors)
                       & offset.between(0, cutoff)].copy()
    accepted["offset_verified"] = offset.loc[accepted.index]
    assert accepted.offset_verified.le(cutoff).all()
    assert accepted.offset_verified.ge(0).all()
    ids = pd.Index(labels.unique_id.astype(str), name="unique_id")
    result = pd.DataFrame(index=ids)
    result.index.name = "unique_id"
    for sensor, names in (("S1", S1), ("S2", S2)):
        if sensor not in sensors:
            continue
        part = accepted.loc[accepted.sensor.eq(sensor)]
        if len(part):
            groups = part.groupby("unique_id")
            # latest uses real image_date; path only breaks ties deterministically
            tie = "npz_relative_path" if "npz_relative_path" in part else "image_date"
            newest = (part.sort_values(["offset_verified", "image_date", tie],
                                       kind="mergesort")
                      .groupby("unique_id").tail(1).set_index("unique_id"))
            med = groups[list(names)].median()
        for variable in names:
            for kind in ("median", "latest"):
                name = f"{variable}__{kind}"
                if not len(part):
                    result[name] = np.nan
                elif kind == "median":
                    result[name] = med[variable].reindex(ids)
                else:
                    result[name] = newest[variable].reindex(ids)
    result = result.reindex(columns=list(SAT_COLUMNS))
    if list(result.index) != list(labels.unique_id.astype(str)):
        raise AssertionError("Cutoff features misaligned with input plot-seasons")
    assert not any("harvest" in c or "relative" in c or "yield" in c
                   for c in result.columns)
    return result.reset_index(drop=True)


def eligible_at_day(stage: pd.DataFrame, day: int) -> pd.Series:
    sow = pd.to_datetime(stage.sowing_date, errors="coerce")
    harvest = pd.to_datetime(stage.harvesting_date, errors="coerce")
    if sow.isna().any() or harvest.isna().any():
        raise ValueError("Missing sowing or harvesting dates")
    duration = (harvest - sow).dt.days
    if (duration <= 0).any():
        raise ValueError("Invalid, nonpositive growing-season duration")
    return duration.gt(day)


def controls_and_satellites(stage: pd.DataFrame, satellite: pd.DataFrame) -> pd.DataFrame:
    if len(stage) != len(satellite):
        raise ValueError("Satellite and survey rows must align")
    x = stage[list((*C_NUM, *C_CAT))].reset_index(drop=True).copy()
    for col in C_NUM:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    for col in C_CAT:
        x[col] = x[col].fillna("__unknown__").astype(str)
    return pd.concat((x, satellite.reset_index(drop=True)), axis=1)


def processed_controls():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median",
                                                  keep_empty_features=True)),
                           ("scale", StandardScaler())]), list(C_NUM)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         list(C_CAT))], sparse_threshold=0)


def processed_full(satellite: bool = True, pca: bool = False):
    steps = [("num", Pipeline([("impute", SimpleImputer(strategy="median",
                                                   keep_empty_features=True)),
                                ("scale", StandardScaler())]), list(C_NUM)),
             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
              list(C_CAT))]
    if satellite:
        last = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler())]
        if pca:
            last.append(("pca", PCA(svd_solver="full")))
        steps.append(("sat", Pipeline(last), list(SAT_COLUMNS)))
    return ColumnTransformer(steps, sparse_threshold=0)


class PartialElasticNet(BaseEstimator, RegressorMixin):
    """Partial penalty via FWL: control coefficients exactly unpenalized.

    Objective on fold-local normalized y and satellite columns:
      min_gamma,beta ||y - C gamma - S beta||²/(2n)
                       + alpha*r*||beta||_1
                       + alpha*(1-r)*||beta||²/2

    C includes intercept, area, year, one-hot season and variety. S contains
    only cutoff-safe satellite summaries; imputation and scaling fitted within
    each cross-validation training fold.
    """
    def __init__(self, alpha: float = 0.03, l1_ratio: float = 0.6):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X: pd.DataFrame, y) -> "PartialElasticNet":
        self.controls_ = processed_controls()
        self.sats_ = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])
        C = self.controls_.fit_transform(X)
        S = self.sats_.fit_transform(X[list(SAT_COLUMNS)])
        C = np.column_stack((np.ones(len(C)), np.asarray(C, float)))
        S = np.asarray(S, float)
        yy = np.asarray(y, float)
        self.ymean_ = float(yy.mean())
        self.ysd_ = max(float(yy.std()), 1e-8)
        ys = (yy - self.ymean_) / self.ysd_
        coef = np.linalg.lstsq(C, np.column_stack((ys, S)), rcond=1e-10)[0]
        rc = ys - C @ coef[:, 0]
        rs = S - C @ coef[:, 1:]
        self.penalized_ = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                                     fit_intercept=False, max_iter=15000, tol=1e-5)
        self.penalized_.fit(rs, rc)
        self.beta_ = self.penalized_.coef_.copy()
        self.gamma_ = np.linalg.lstsq(C, ys - S @ self.beta_, rcond=1e-10)[0]
        return self

    def predict(self, X: pd.DataFrame):
        C = self.controls_.transform(X)
        C = np.column_stack((np.ones(len(C)), np.asarray(C, float)))
        S = self.sats_.transform(X[list(SAT_COLUMNS)])
        return self.ymean_ + self.ysd_ * (C @ self.gamma_ + S @ self.beta_)


def model_grid(name: str):
    if name == "ridge_controls":
        return Pipeline([("prep", processed_full(satellite=False)),
                         ("model", Ridge())]), {"model__alpha": [1., 10., 100., 1000.]}
    if name == "ridge_s1s2":
        return Pipeline([("prep", processed_full()),
                         ("model", Ridge())]), {"model__alpha": [10., 100., 300., 1000.]}
    if name == "pca_ridge_s1s2":
        return Pipeline([("prep", processed_full(pca=True)),
                         ("model", Ridge())]), {
            "prep__sat__pca__n_components": [2, 4, 6],
            "model__alpha": [10., 100., 1000.]}
    if name == "partial_elastic_s1s2":
        return PartialElasticNet(), {"alpha": [0.01, 0.03, 0.1, 0.3],
                                    "l1_ratio": [0.5, 0.8]}
    if name == "extra_trees_controls":
        return Pipeline([("prep", processed_full(satellite=False)),
                         ("model", ExtraTreesRegressor(
                             n_estimators=160, random_state=2026, n_jobs=2))]), {
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features": [0.7, 1.0]}
    raise KeyError(name)


def tune_predict(train_x: pd.DataFrame, train_y, groups, test_x: pd.DataFrame,
                 model: str):
    pipeline, grid = model_grid(model)
    if pd.Series(groups).nunique() < 3:
        raise ValueError("At least three training plots required for nested CV")
    search = GridSearchCV(pipeline, grid, scoring="neg_mean_squared_error",
                          cv=GroupKFold(n_splits=3), n_jobs=1, refit=True,
                          error_score="raise")
    search.fit(train_x, np.asarray(train_y, float), groups=np.asarray(groups))
    return search.predict(test_x), search.best_params_


def metrics(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    return {"RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "MAE": float(mean_absolute_error(y, pred)),
            "R2": float(r2_score(y, pred))}


def paired_plot_interval(y, plots, reference, candidate, reps=1000,
                         seed=2026):
    y, plots = np.asarray(y, float), np.asarray(plots)
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    unique = np.unique(plots)
    index = {g: np.flatnonzero(plots == g) for g in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(reps):
        selected = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([index[g] for g in selected])
        samples.append(metrics(y[idx], reference[idx])["RMSE"] -
                       metrics(y[idx], candidate[idx])["RMSE"])
    return np.quantile(samples, [.025, .975]).tolist()


def phase1_audit(stage: pd.DataFrame, observations: pd.DataFrame,
                 verify_days=True):
    """Only TRAIN records used for QA results; never VAL outcomes."""
    before = len(stage)
    if stage.unique_id.duplicated().any():
        raise ValueError("Duplicate plot-season ID")
    if not pd.to_numeric(stage['yield'], errors='coerce').gt(0).all():
        raise ValueError("Invalid, missing or nonpositive training yield")
    durations = (pd.to_datetime(stage.harvesting_date) -
                 pd.to_datetime(stage.sowing_date)).dt.days
    if durations.isna().any() or durations.min() <= 0:
        raise ValueError("Invalid crop growing-season dates")
    if not set(observations.unique_id).issubset(set(stage.unique_id)):
        raise ValueError("Observation file must be restricted to TRAIN IDs before audit")
    timing = verify_observation_times(observations, stage) if verify_days else {}
    rows = []
    for day in CUTS:
        eligible = int((durations > day).sum())
        rows.append({"cutoff_days": day, "eligible_rows": eligible,
                     "excluded_harvested_rows": before - eligible,
                     "minimum_days_remaining": (int(durations[durations > day].min()-day)
                                                if eligible else None)})
    missing = []
    obs = observations.loc[observations.status.eq('ok')]
    for sensor, names in (("S1", S1), ("S2", S2)):
        seen = obs.loc[obs.sensor.eq(sensor)]
        for feat in names:
            missing.append({"sensor": sensor, "variable": feat,
                            "accepted_scene_count": len(seen),
                            "missing_feature_fraction": float(seen[feat].isna().mean())
                            if len(seen) else 1.})
    result = {"train_rows": int(before), "physical_plots": int(stage.plot_id.nunique()),
              "years": sorted(stage.year.unique().tolist()),
              "min_growing_days": int(durations.min()),
              "max_growing_days": int(durations.max()),
              "observations": timing, "yield_units": "kg/acre",
              "yield_units_source": SOURCE_NOTE,
              "target": "farmer-recalled per-acre yield, not measured harvest",
              "notes": ["Yield unit from original SICKLE supplementary Figure 4d",
                        "Coordinates and ground-survey accuracy not independently audited",
                        "No claim of prospective forecast for retrospective phenology windows"]}
    return result, pd.DataFrame(rows), pd.DataFrame(missing)


def training_and_observations():
    train = load_stage(train_only=True)
    raw = load_obs()
    # Never give stage/observation val rows to training-audit or selection.
    raw = raw.loc[raw.unique_id.isin(train.unique_id)].copy()
    return train, raw


def audit_run() -> None:
    train, obs = training_and_observations()
    audit, eligibility, missing = phase1_audit(train, obs)
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / "01_data_timing_audit.json", audit)
    eligibility.to_csv(OUT / "01_cutoff_eligibility.csv", index=False)
    missing.to_csv(OUT / "01_scene_missingness.csv", index=False)
    print("DATA AUDIT:", audit, flush=True)


def cutoff_run() -> None:
    """Common 90-day cohort, day45 primary; 5x3 group-nested CV."""
    train, obs = training_and_observations()
    phase1_audit(train, obs)  # hard blocker before any cutoff experiment
    eligible = eligible_at_day(train, max(CUTS))
    work = train.loc[eligible].reset_index(drop=True)
    assert len(work) > 0
    ys = work['yield'].to_numpy(float)
    groups = work.plot_id.to_numpy()
    folds = list(GroupKFold(n_splits=5).split(work, ys, groups))
    predictions = work[["unique_id", "plot_id", "yield", "year"]].copy()
    choices, results, observations_used = [], [], []
    for day in CUTS:
        sat = cutoff_features(obs, work, day)
        frame = controls_and_satellites(work, sat)
        for foldno, (i_tr, i_te) in enumerate(folds, 1):
            assert set(groups[i_tr]).isdisjoint(groups[i_te])
        for model in CORE_MODELS:
            pred = np.full(len(work), np.nan)
            for foldno, (i_tr, i_te) in enumerate(folds, 1):
                prediction, best = tune_predict(frame.iloc[i_tr], ys[i_tr],
                                                groups[i_tr], frame.iloc[i_te], model)
                pred[i_te] = prediction
                choices.append({"day": day, "model": model, "outer_fold": foldno,
                                "training_rows": len(i_tr), "heldout_rows": len(i_te),
                                "tuning": best})
            assert np.isfinite(pred).all()
            name = f"day_{day:03d}__{model}"
            predictions[name] = pred
            result = {"cutoff_days": day, "model": model, "num_plots": work.plot_id.nunique(),
                      "num_plot_seasons": len(work), "primary_day": day == PRIMARY_DAY,
                      **metrics(ys, pred)}
            ref = predictions[f"day_{day:03d}__ridge_controls"]
            result['RMSE_controls_minus_model'] = (
                metrics(ys, ref)["RMSE"] - result['RMSE'])
            result['conditional_bootstrap_low'], result['conditional_bootstrap_high'] = (
                paired_plot_interval(ys, groups, ref, pred, reps=600, seed=2026+day))
            results.append(result)
            print('CUTOFF', day, model, result['RMSE'], flush=True)
        observed_frac = sat.notna().mean()
        for col, fraction in observed_frac.items():
            observations_used.append({"day": day, "feature": col,
                                      "observed_fraction": float(fraction)})
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT / "02_cutoff_grouped_oof.csv", index=False)
    pd.DataFrame(results).to_csv(OUT / "02_cutoff_grouped_cv.csv", index=False)
    pd.DataFrame(choices).to_csv(OUT / "02_nested_tuning.csv", index=False)
    pd.DataFrame(observations_used).to_csv(OUT / "02_cutoff_coverage.csv", index=False)
    save_json(OUT / "02_cutoff_configuration.json", {
        "cutoffs": CUTS, "primary_day": PRIMARY_DAY, "shared_eligible_cohort": True,
        "n_plots": int(work.plot_id.nunique()), "n_rows": len(work),
        "models": CORE_MODELS, "outer_folds": 5, "inner_folds": 3,
        "validation_scored": False,
        "cautions": ["Repeated historical TRAIN plots and model exploration",
                     "CI conditional on OOF predictions; no refit or search adjustment",
                     "Only known sowing date, date, and input sensor values used"]})



def transfer_run() -> None:
    """Matched plot, source-year to target-year transfer at PRIMARY_DAY.

    A single training fit predicts source-year and other-year records of the
    *same* held-out plots. This isolates the temporal transfer component from
    the earlier much-larger pooled-year training samples. Still not causal:
    source and target outcomes are distinct plot-seasons.
    """
    train, obs = training_and_observations()
    phase1_audit(train, obs)
    good = eligible_at_day(train, max(CUTS))
    work = train.loc[good].reset_index(drop=True)
    sat = cutoff_features(obs, work, PRIMARY_DAY)
    frame = controls_and_satellites(work, sat)
    indexed = pd.concat((work[["unique_id", "plot_id", "yield"]],
                         frame), axis=1)
    records, choices = [], []
    for source_year, target_year in ((2019, 2020), (2020, 2019)):
        source = indexed.loc[indexed.year.eq(source_year)]
        target = indexed.loc[indexed.year.eq(target_year)]
        split = GroupKFold(n_splits=5).split(target, groups=target.plot_id)
        seen_source, seen_target = set(), set()
        for fold, (_, te) in enumerate(split, start=1):
            outside = target.iloc[te]
            testplots = set(outside.plot_id)
            same = source.loc[source.plot_id.isin(testplots)]
            fit = source.loc[~source.plot_id.isin(testplots)]
            assert len(fit) and len(same) and len(outside)
            assert set(same.plot_id) == testplots
            assert not set(fit.plot_id) & testplots
            assert not seen_source & set(same.unique_id)
            assert not seen_target & set(outside.unique_id)
            seen_source.update(same.unique_id)
            seen_target.update(outside.unique_id)
            test = pd.concat((same, outside), ignore_index=True)
            predictions = test[["unique_id", "plot_id", "year", "yield"]].copy()
            predictions["train_year"] = source_year
            predictions["outer_fold"] = fold
            predictions["evaluation"] = (["within_year"] * len(same)
                                         + ["cross_year"] * len(outside))
            predictions["source_training_rows"] = len(fit)
            # Year is constant within a source-year training split; explicitly
            # remove it as a predictor for temporal transfer, rather than
            # applying the future calendar year's numeric value at test time.
            columns = list((*C_NUM, *C_CAT, *SAT_COLUMNS))
            source_x = fit[columns].copy()
            target_x = test[columns].copy()
            source_x.loc[:, "year"] = 0
            target_x.loc[:, "year"] = 0
            for model in CORE_MODELS:
                pred, best = tune_predict(source_x, fit['yield'].to_numpy(float),
                                          fit.plot_id.to_numpy(), target_x,
                                          model)
                predictions[model] = pred
                choices.append({"source_year":source_year,"fold":fold,
                                "model":model,"best_params":best})
            records.append(predictions)
            print(f"TRANSFER {source_year}->{target_year} fold {fold}/5", flush=True)
        assert set(source.unique_id) == seen_source
        assert set(target.unique_id) == seen_target
    pred = pd.concat(records, ignore_index=True)
    assert len(pred) == 2 * len(indexed)
    summaries = []
    for year in (2019, 2020):
        for evaluation in ("within_year", "cross_year"):
            sub = pred.loc[pred.train_year.eq(year)
                           & pred.evaluation.eq(evaluation)]
            for model in CORE_MODELS:
                summaries.append({"training_year":year, "evaluation": evaluation,
                                  "target_year":int(sub.year.iloc[0]),
                                  "model":model,
                                  "test_rows":len(sub),
                                  "test_distinct_plots":sub.plot_id.nunique(),
                                  **metrics(sub['yield'],sub[model])})
    OUT.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT / "04_cutoff_day45_transfer_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "04_cutoff_day45_transfer_metrics.csv",
                                    index=False)
    pd.DataFrame(choices).to_csv(OUT / "04_cutoff_day45_transfer_tuning.csv",
                                   index=False)
    save_json(OUT / "04_cutoff_day45_transfer_configuration.json", {
        "primary_day": PRIMARY_DAY, "source_training_years": [2019, 2020],
        "same_fitted_model_predicts_both_years": True,
        "training_test_physical_plots_disjoint": True,
        "hyperparameters_chosen_only_from_source_training_plots": True,
        "models": CORE_MODELS, "outcomes_are_distinct_plot_seasons": True,
        "official_val_scored": False})


def panel_pairs(train: pd.DataFrame, require_same_variety=True):
    """Defensibly unique (plot, season, year) paired observations.

    Exclude ambiguous duplicates; match same variety by default. Never pair
    different standard seasons or choose records based on the yield value.
    """
    unique_years = sorted(train.year.unique())
    if unique_years != [2019, 2020]:
        raise ValueError("Expected the 2019/2020 matched panel")
    keep = train.loc[train.year.isin(unique_years)].copy()
    for col in ("season", "variety"):
        keep[col] = keep[col].fillna("__unknown__").astype(str)
    count = keep.groupby(["plot_id", "season", "year"], dropna=False).size()
    valid = count.loc[count.eq(1)].reset_index()[["plot_id", "season", "year"]]
    uniq = keep.merge(valid, on=["plot_id", "season", "year"],
                      how="inner", validate="many_to_one")
    assert not uniq.duplicated(["plot_id", "season", "year"]).any()
    p0 = uniq.loc[uniq.year.eq(unique_years[0])]
    p1 = uniq.loc[uniq.year.eq(unique_years[1])]
    paired = p0.merge(p1, on=["plot_id", "season"], how="inner",
                      suffixes=("_2019", "_2020"), validate="one_to_one")
    paired['same_variety'] = paired.variety_2019.eq(paired.variety_2020)
    # Standard-season categories can span months. Only match crop cycles
    # with comparable annual sowing dates, independently of yield.
    first_sow = pd.to_datetime(paired.sowing_date_2019, errors="coerce")
    second_sow = pd.to_datetime(paired.sowing_date_2020, errors="coerce")
    gap = (first_sow.dt.dayofyear - second_sow.dt.dayofyear).abs()
    paired['circular_sowing_gap_days'] = np.minimum(gap, 366 - gap)
    paired = paired.loc[paired.circular_sowing_gap_days.le(
        PAIR_SOWING_MAX_CIRCULAR_DAYS)].copy()
    if require_same_variety:
        paired = paired.loc[paired.same_variety].copy()
    paired['delta_yield'] = (paired['yield_2020'].to_numpy(float) -
                             paired['yield_2019'].to_numpy(float))
    for feature in ("W3_40_60__s2_ndvi", "W3_40_60__s2_ndmi",
                    "W3_40_60__s1_vv_db", "W3_40_60__s1_vv_minus_vh_db"):
        if feature in train.columns:
            paired[f"delta__{feature}"] = (
                pd.to_numeric(paired[f"{feature}_2020"], errors="coerce") -
                pd.to_numeric(paired[f"{feature}_2019"], errors="coerce"))
    return paired


def panel_run() -> None:
    train = load_stage(train_only=True)  # never read official VAL yield
    loose = panel_pairs(train, require_same_variety=False)
    strict = panel_pairs(train, require_same_variety=True)
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {"unrestricted_unique_plot_season_pairs": len(loose),
             "strict_same_variety_pairs": len(strict),
             "strict_distinct_physical_plots": int(strict.plot_id.nunique()),
             "matched_2019_2020_seasons_only": True,
             "maximum_circular_sowing_date_gap_days": PAIR_SOWING_MAX_CIRCULAR_DAYS,
             "unmatched_seasons_and_ambiguous_duplicates_excluded": True,
             "panel_is_not_causal": True,
             "regression_status": "pending feasibility check"}
    model_data = [x for x in strict.columns if x.startswith("delta__")]
    minimum = 30
    if strict.plot_id.nunique() >= minimum and len(model_data) >= 2:
        # Panel outcome ΔY is for comparison of two *known* seasons, not a
        # prediction on genuinely new plots when Y_previous is unknown.
        y = strict.delta_yield.to_numpy(float)
        group = strict.plot_id.to_numpy()
        X = strict[model_data]
        folds = list(GroupKFold(n_splits=5).split(strict, y, group))
        models = {"mean_delta": np.full(len(strict), np.nan),
                  "ridge_delta_satellites": np.full(len(strict), np.nan)}
        for tr, te in folds:
            models['mean_delta'][te] = y[tr].mean()
            model = Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()), ("ridge", Ridge())])
            cv = GridSearchCV(model, {"ridge__alpha": [10., 100., 1000.]},
                              cv=GroupKFold(n_splits=3),
                              scoring="neg_mean_squared_error")
            cv.fit(X.iloc[tr], y[tr], groups=group[tr])
            models['ridge_delta_satellites'][te] = cv.predict(X.iloc[te])
        prediction = strict[["plot_id", "delta_yield"]].copy()
        for label, values in models.items():
            prediction[label] = values
        prediction.to_csv(OUT / "03_panel_difference_oof.csv", index=False)
        summary = [{"model": k, "n_pairs": len(strict), "n_plots":len(np.unique(group)),
                    **metrics(y, v)} for k, v in models.items()]
        pd.DataFrame(summary).to_csv(OUT / "03_panel_difference_metrics.csv", index=False)
        audit['regression_status'] = "completed grouped plot-level CV"
    else:
        audit['regression_status'] = (f"descriptive only; required >= {minimum} "
                                      "physical plots and >= 2 delta features")
    # Prevent labels inadvertently exposed in a reusable project data extract:
    # persist only IDs and matching diagnostics, not private yield labels.
    pd.DataFrame({"plot_id":strict.plot_id,"season":strict.season,
                  "same_variety":strict.same_variety,
                  "circular_sowing_gap_days":strict.circular_sowing_gap_days}).to_csv(
        OUT / "03_panel_match_audit.csv", index=False)
    save_json(OUT / "03_panel_feasibility.json", audit)
    print('PANEL:', audit, flush=True)


def frozen_manifest(output_dir: Path | None = None) -> dict:
    """Requires completed exploratory experiments; freeze code + data hashes."""
    if output_dir is None:
        output_dir = OUT
    required = (output_dir / "01_data_timing_audit.json",
                output_dir / "02_cutoff_grouped_cv.csv",
                output_dir / "03_panel_feasibility.json",
                output_dir / "04_cutoff_day45_transfer_metrics.csv")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Complete earlier phases before freeze: {path}")
    audit = json.loads(required[0].read_text())
    assert audit["train_rows"] == 282
    return {"protocol_version": "EML-SICKEL-roadmap-v1",
            "predeclared_primary_cutoff": PRIMARY_DAY,
            "frozen_final_models": FINAL_MODELS,
            "primary_comparison": "pca_ridge_s1s2 versus ridge_controls at day45",
            "guardrail": "Do not change code/models/hyperparameters after evaluating official VAL",
            "prior_validation_exposure": (
                "Official VAL was previously inspected for Ridge; this is a reserved "
                "comparison, NOT a pristine independent external validation"),
            "stage_sha256": sha256(STAGE_FILE),
            "observations_sha256": sha256(OBS_FILE),
            "source_sha256": sha256(Path(__file__).resolve()),
            "csv_report_sha256": sha256(required[1]),
            "transfer_sha256": sha256(required[3]),
            "yield_unit": "kg/acre",
            "environment": {"numpy": np.__version__, "pandas": pd.__version__,
                            "scikit_learn": sklearn.__version__}}


def freeze_run() -> None:
    outfile = OUT / "04_frozen_manifest.json"
    if outfile.exists():
        raise FileExistsError(f"Manifest already exists; do not silently re-freeze: {outfile}")
    save_json(outfile, frozen_manifest())
    print('Frozen candidate list and data/code hashes:', outfile, flush=True)


def validation_run() -> None:
    """One explicit reserved comparison; never tune using VAL outcomes."""
    frozen = OUT / "04_frozen_manifest.json"
    done = OUT / "05_final_validation_comparison.csv"
    attempted = OUT / "05_validation_attempted.json"
    if done.exists() or attempted.exists():
        raise FileExistsError("Final validation already attempted; do not evaluate repeatedly")
    if not frozen.is_file():
        raise FileNotFoundError("Freeze model and code hashes first")
    expected = json.loads(frozen.read_text())
    current = frozen_manifest()
    for name in ("stage_sha256", "observations_sha256", "source_sha256",
                 "csv_report_sha256", "transfer_sha256", "frozen_final_models",
                 "predeclared_primary_cutoff", "environment"):
        # JSON restores frozen tuples as lists. Compare actual model names,
        # not the container types, while preserving all other equality checks.
        actual = list(current[name]) if name == "frozen_final_models" else current[name]
        if expected[name] != actual:
            raise RuntimeError(f"Frozen manifest mismatch in {name}; no final scoring")
    # Record a one-time attempt *before* reading any reserved outcomes.
    # A failed attempt needs manual reconciliation, not silent re-scoring.
    save_json(attempted, {"manifest": str(frozen),
                          "prior_val_exposure_disclosed": True,
                          "automatic_retries_permitted": False})
    full = load_stage(train_only=False)
    train = full.loc[full.split.eq("train")].reset_index(drop=True)
    val = full.loc[full.split.eq("val")].reset_index(drop=True)
    obs = load_obs()
    # We can read reserved VAL records only from here onwards.
    selected = pd.concat((train, val), ignore_index=True)
    if not eligible_at_day(selected, PRIMARY_DAY).all():
        raise RuntimeError("Some records harvested before primary cutoff")
    fullsat = cutoff_features(obs, selected, PRIMARY_DAY)
    all_x = controls_and_satellites(selected, fullsat)
    n = len(train)
    x_train = all_x.iloc[:n]
    x_val = all_x.iloc[n:]
    y_train, y_val = train['yield'].to_numpy(float), val['yield'].to_numpy(float)
    predictions = {"unique_id": val.unique_id, "plot_id": val.plot_id,
                   "yield": y_val}
    choices = {}
    results = []
    for name in FINAL_MODELS:
        pred, best = tune_predict(x_train, y_train, train.plot_id.to_numpy(),
                                  x_val, name)
        choices[name] = best
        predictions[name] = pred
        results.append({"model": name, "n_val_rows": len(val),
                        "n_val_plots": val.plot_id.nunique(),
                        **metrics(y_val, pred)})
    pd.DataFrame(predictions).to_csv(OUT / "05_final_validation_predictions.csv", index=False)
    pd.DataFrame(results).to_csv(done, index=False)
    save_json(OUT / "05_final_tuning_on_training_only.json", choices)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


def main(args=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "cutoffs", "panel",
                                            "transfer", "freeze", "validate"))
    opt = parser.parse_args(args)
    commands = {"audit": audit_run, "cutoffs": cutoff_run,
                "panel": panel_run, "transfer": transfer_run, "freeze": freeze_run,
                "validate": validation_run}
    commands[opt.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
