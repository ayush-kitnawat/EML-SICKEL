"""Synthetic smoke/leakage checks. They are NOT reported empirical results."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import research_roadmap as r


def synthetic(n_plots=40, seed=3):
    rng = np.random.default_rng(seed)
    records = []
    scenes = []
    for i in range(n_plots):
        for year in (2019, 2020):
            unique_id = f"{i:04d}-{year}"
            sow = pd.Timestamp(f"{year}-06-01")
            harvest = sow + pd.Timedelta(days=125)
            records.append(dict(unique_id=unique_id, plot_id=f"P{i:03d}",
                                split="train", **{"yield":float(1700 + 2*i + year-2019 +rng.normal(0, 120))},
                                area=float(0.7 + (i % 3)*.15), year=year,
                                variety="BPT" if i % 2 else "CR", season="may-sep",
                                sowing_date=str(sow.date()),
                                harvesting_date=str(harvest.date()),
                                **{f"W3_40_60__{f}": float(rng.normal()) for f in (
                                    "s2_ndvi","s2_ndmi","s1_vv_db","s1_vv_minus_vh_db")}))
            for sensor in ("S1", "S2"):
                for day in (8, 26, 43, 61, 94):
                    row = dict(unique_id=unique_id, sensor=sensor, status="ok",
                               image_date=str((sow + pd.Timedelta(days=day)).date()),
                               days_since_sowing=day,
                               npz_relative_path=f"{unique_id}/{sensor}/{day}.npz",
                               relative_season_time=day / 125)
                    for feature in r.SATELLITE_FEATURES:
                        row[feature] = float(rng.normal()) if feature in (
                            r.S1 if sensor == 'S1' else r.S2) else np.nan
                    scenes.append(row)
    return pd.DataFrame(records), pd.DataFrame(scenes)


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage, cls.obs = synthetic()

    def test_cutoff_is_independent_of_post_cutoff_acquisitions(self):
        initial = r.cutoff_features(self.obs, self.stage, 45)
        changed = self.obs.copy()
        # Assert no post-cutoff scene can alter a model feature value.
        post = changed.days_since_sowing.gt(45)
        for feat in r.SATELLITE_FEATURES:
            changed.loc[post, feat] = 1e10
        observed = r.cutoff_features(changed, self.stage, 45)
        pd.testing.assert_frame_equal(initial, observed, check_exact=True)

    def test_cutoff_is_independent_of_harvest_date_and_label(self):
        frame = self.stage.copy()
        first = r.cutoff_features(self.obs, frame, 45)
        frame.loc[:, 'harvesting_date'] = '2024-01-01'
        frame.loc[:, 'yield'] = -9999.
        second = r.cutoff_features(self.obs, frame, 45)
        pd.testing.assert_frame_equal(first, second, check_exact=True)

    def test_cutoff_uses_latest_before_day_and_real_acquisition_date(self):
        single = self.stage.iloc[:1]
        scenes = self.obs.loc[self.obs.unique_id.eq(single.unique_id.iloc[0])]
        sat = r.cutoff_features(scenes, single, 45)
        self.assertTrue(pd.notna(sat.iloc[0]['s2_ndvi__median']))
        self.assertNotEqual(sat.iloc[0]['s2_ndvi__median'], sat.iloc[0]['s2_ndvi__latest'])
        # 43-day scene must be the latest; 61-day one must never be included.
        most_recent = scenes.loc[scenes.sensor.eq('S2') & scenes.days_since_sowing.eq(43)]
        self.assertAlmostEqual(sat.iloc[0]['s2_ndvi__latest'],most_recent.s2_ndvi.iloc[0])
        self.assertEqual(list(sat.columns), list(r.SAT_COLUMNS))

    def test_untrustworthy_precomputed_days_raise(self):
        tampered = self.obs.copy()
        tampered.loc[tampered.index[0], 'days_since_sowing'] = -5
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            r.verify_observation_times(tampered, self.stage)

    def test_ineligible_day_rejected(self):
        stage = self.stage.copy()
        stage.loc[stage.index[0], 'harvesting_date'] = '2019-06-30'
        eligible = r.eligible_at_day(stage, 45)
        self.assertFalse(eligible.iloc[0])
        self.assertTrue(eligible.iloc[1:].all())

    def test_model_covariates_exclude_harvest_and_yield(self):
        sat = r.cutoff_features(self.obs, self.stage, 45)
        X = r.controls_and_satellites(self.stage, sat)
        self.assertEqual(set(X.columns), set(r.C_NUM+r.C_CAT+r.SAT_COLUMNS))
        for disallowed in ('harvesting_date','sowing_date','yield','relative_season_time'):
            self.assertNotIn(disallowed, X)

    def test_partial_elastic_controls_unpenalized_mathematically(self):
        # Compare FWL with unpenalized OLS on survey controls as alpha tends high.
        satellite = r.cutoff_features(self.obs, self.stage, 30)
        X = r.controls_and_satellites(self.stage, satellite)
        y = self.stage['yield'].to_numpy(float)
        m = r.PartialElasticNet(alpha=100., l1_ratio=.9).fit(X, y)
        self.assertLess(np.max(abs(m.beta_)), 1e-4)
        C = m.controls_.transform(X)
        C = np.column_stack((np.ones(len(C)), np.asarray(C, float)))
        ols_y = np.linalg.lstsq(C, y, rcond=1e-10)[0]
        np.testing.assert_allclose(m.predict(X), C @ ols_y, atol=1e-4)

    def test_nested_training_models_fit_predict_and_yield_finite(self):
        sat = r.cutoff_features(self.obs, self.stage, 45)
        X = r.controls_and_satellites(self.stage, sat)
        for name in r.CORE_MODELS:
            with self.subTest(name=name):
                pred, best = r.tune_predict(X.iloc[10:],
                                            self.stage['yield'].iloc[10:],
                                            self.stage.plot_id.iloc[10:],
                                            X.iloc[:10], name)
                self.assertEqual(len(pred), 10)
                self.assertTrue(np.isfinite(pred).all())
                self.assertTrue(best)

    def test_panel_strict_pairing_excludes_variety_switches(self):
        panel = r.panel_pairs(self.stage)
        self.assertEqual(len(panel), 40)
        self.assertTrue(panel.same_variety.all())
        stage = self.stage.copy()
        stage.loc[stage.unique_id.eq('0001-2020'), 'variety'] = 'DIFFERENT'
        panel2 = r.panel_pairs(stage)
        self.assertEqual(len(panel2), 39)
        self.assertTrue(set(panel2.plot_id).issubset(set(panel.plot_id)))

    def test_panel_sowing_mismatch_not_used(self):
        frame = self.stage.copy()
        frame.loc[frame.unique_id.eq('0002-2020'), 'sowing_date'] = '2020-11-15'
        matched = r.panel_pairs(frame)
        self.assertEqual(len(matched), 39)

    def test_panel_duplicate_season_excluded_not_cherry_picked(self):
        stage = pd.concat([self.stage, self.stage.iloc[[0]].assign(
            unique_id='duplicate-2019')], ignore_index=True)
        pair = r.panel_pairs(stage)
        self.assertEqual(len(pair), 39)

    def test_data_audit_properties(self):
        a, eligibility, coverage = r.phase1_audit(self.stage, self.obs)
        self.assertEqual(a['physical_plots'], 40)
        self.assertEqual(a['yield_units'], 'kg/acre')
        self.assertEqual(eligibility.eligible_rows.tolist(), [80]*6)
        self.assertEqual(len(coverage), len(r.SATELLITE_FEATURES))


if __name__ == '__main__':
    unittest.main()

class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage, cls.obs = synthetic(n_plots=24)

    def test_audit_cutoff_and_panel_commands_on_synthetic_data(self):
        from unittest.mock import patch
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch.object(r, 'OUT', output), \
                 patch.object(r, 'training_and_observations',
                              lambda:(self.stage, self.obs)), \
                 patch.object(r, 'load_stage',
                              lambda train_only=True:self.stage), \
                 patch.object(r, 'CUTS', (45,)), \
                 patch.object(r, 'CORE_MODELS',
                              ('ridge_controls', 'partial_elastic_s1s2')):
                r.audit_run()
                r.cutoff_run()
                r.panel_run()
                self.assertEqual(len(pd.read_csv(output/'02_cutoff_grouped_cv.csv')), 2)
                self.assertEqual(len(pd.read_csv(output/'02_cutoff_grouped_oof.csv')), 48)
                self.assertEqual(len(pd.read_csv(output/'03_panel_match_audit.csv')), 24)
                self.assertEqual(len(pd.read_csv(output/'01_cutoff_eligibility.csv')), 1)

    def test_match_year_transfer_on_synthetic_data(self):
        from unittest.mock import patch
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            with patch.object(r, 'OUT', output), \
                 patch.object(r, 'training_and_observations',
                              lambda:(self.stage, self.obs)), \
                 patch.object(r,'CORE_MODELS',('ridge_controls',)), \
                 patch.object(r,'CUTS',(45,)):
                r.transfer_run()
                preds=pd.read_csv(output/'04_cutoff_day45_transfer_predictions.csv')
                self.assertEqual(len(preds), 2*len(self.stage))
                self.assertFalse(preds[['train_year','evaluation','unique_id']].duplicated().any())
                for year in (2019,2020):
                    within=preds.loc[preds.train_year.eq(year) & preds.evaluation.eq('within_year')]
                    other=preds.loc[preds.train_year.eq(year) & preds.evaluation.eq('cross_year')]
                    self.assertEqual(set(within.plot_id),set(other.plot_id))
                    self.assertEqual(within.plot_id.nunique(),24)
                    self.assertTrue(np.isfinite(within.ridge_controls).all())

    def test_freeze_is_one_way_and_detects_input_drift_before_val(self):
        from unittest.mock import patch
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            fake_stage=output/'stage.csv'
            fake_obs=output/'obs.csv'
            fake_stage.write_text('header\nfirst\n')
            fake_obs.write_text('header\nfirst\n')
            r.save_json(output/'01_data_timing_audit.json',{'train_rows':282})
            (output/'02_cutoff_grouped_cv.csv').write_text('model,RMSE\nridge_controls,100\n')
            (output/'04_cutoff_day45_transfer_metrics.csv').write_text('model,RMSE\nridge_controls,200\n')
            r.save_json(output/'03_panel_feasibility.json',{'regression_status':'descriptive'})
            with patch.object(r,'OUT',output),patch.object(r,'STAGE_FILE',fake_stage), \
                 patch.object(r,'OBS_FILE',fake_obs):
                r.freeze_run()
                with self.assertRaises(FileExistsError):
                    r.freeze_run()
                fake_stage.write_text('header\nCHANGED\n')
                with self.assertRaisesRegex(RuntimeError,'Frozen manifest mismatch'):
                    r.validation_run()
                self.assertFalse((output/'05_final_validation_comparison.csv').exists())

    def test_frozen_json_model_list_passes_preflight_without_reading_val(self):
        """JSON converts the frozen tuple of model names to a list."""
        from unittest.mock import patch
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            fake_stage, fake_obs = output/'stage.csv', output/'obs.csv'
            fake_stage.write_text('example')
            fake_obs.write_text('example')
            r.save_json(output/'01_data_timing_audit.json', {'train_rows': 282})
            (output/'02_cutoff_grouped_cv.csv').write_text('model,RMSE\nridge_controls,100\n')
            (output/'04_cutoff_day45_transfer_metrics.csv').write_text(
                'model,RMSE\nridge_controls,200\n')
            r.save_json(output/'03_panel_feasibility.json', {'regression_status': 'complete'})
            class ReachedPostPreflight(Exception):
                pass
            with patch.object(r, 'OUT', output), \
                 patch.object(r, 'STAGE_FILE', fake_stage), \
                 patch.object(r, 'OBS_FILE', fake_obs), \
                 patch.object(r, 'load_stage', side_effect=ReachedPostPreflight):
                r.freeze_run()
                with self.assertRaises(ReachedPostPreflight):
                    r.validation_run()
            self.assertTrue((output/'05_validation_attempted.json').exists())
            self.assertFalse((output/'05_final_validation_comparison.csv').exists())
