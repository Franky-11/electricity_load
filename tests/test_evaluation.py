import numpy as np
import pandas as pd

from power_forecast.evaluation import score_forecast, summarize_scores, walk_forward_folds


def test_walk_forward_folds_uses_common_fold_definition():
    idx = pd.date_range("2026-01-01 00:00", periods=10 * 24, freq="h", tz="Europe/Berlin")
    s = pd.Series(range(len(idx)), index=idx)

    folds = walk_forward_folds(s, H=24, win_days=3, eval_days=2, step_hours=24)

    assert len(folds) == 2
    assert len(folds[0].train) == 72
    assert len(folds[0].test) == 24
    assert folds[0].train.index.tz is None


def test_score_forecast_computes_model_and_baseline_metrics():
    idx = pd.date_range("2026-01-01 00:00", periods=3, freq="h")
    y_true = pd.Series([100.0, 120.0, 140.0], index=idx)
    y_pred = pd.Series([110.0, 115.0, 130.0], index=idx)
    baseline = pd.Series([90.0, 100.0, 120.0], index=idx)
    insample = pd.Series([80.0, 100.0, 120.0, 140.0])

    row = score_forecast(y_true, y_pred, baseline=baseline, insample=insample, mase_m=1)

    assert row["valid"] is True
    assert row["points_compared"] == 3
    assert np.isclose(row["MAE"], 25 / 3)
    assert np.isclose(row["MAE_base"], 50 / 3)
    assert np.isclose(row["Gain"], (row["MAE_base"] - row["MAE"]) / row["MAE_base"] * 100)


def test_score_forecast_marks_low_coverage_invalid():
    idx = pd.date_range("2026-01-01 00:00", periods=4, freq="h")
    y_true = pd.Series([1.0, np.nan, np.nan, np.nan], index=idx)
    y_pred = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)

    row = score_forecast(y_true, y_pred, min_coverage=0.8)

    assert row["valid"] is False
    assert np.isnan(row["MAE"])


def test_score_forecast_evaluates_prediction_interval():
    idx = pd.date_range("2026-01-01 00:00", periods=4, freq="h")
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx)
    y_pred = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx)
    interval = pd.DataFrame({"lo": [5.0, 15.0, 35.0, 35.0], "hi": [15.0, 25.0, 45.0, 45.0]}, index=idx)

    row = score_forecast(y_true, y_pred, interval=interval, nominal_coverage=0.75)

    assert row["PI_coverage_pct"] == 75.0
    assert row["PI_mean_width_MW"] == 10.0
    assert row["PI_calibration_error_pct"] == 0.0


def test_summarize_scores_averages_only_valid_rows():
    summary = summarize_scores([
        {"valid": True, "MAE": 10.0, "sMAPE": 1.0, "MAE_base": 20.0, "points_compared": 24, "expected_points": 24, "coverage_pct": 100.0, "Gain": 50.0},
        {"valid": False, "MAE": np.nan, "sMAPE": np.nan, "MAE_base": np.nan, "points_compared": 1, "expected_points": 24, "coverage_pct": 4.2, "Gain": np.nan},
    ])

    assert summary["folds"].iloc[0] == 1
    assert summary["MAE"].iloc[0] == 10.0
