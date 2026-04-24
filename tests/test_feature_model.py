import numpy as np
import pandas as pd

import power_forecast.feature_model as fm


def _load_series(periods: int = 40 * 24) -> pd.Series:
    idx = pd.date_range("2026-01-01 00:00", periods=periods, freq="h")
    hour = idx.hour.to_numpy()
    week = np.arange(periods) % 168
    values = 50_000 + 2_000 * np.sin(2 * np.pi * hour / 24) + 500 * np.cos(2 * np.pi * week / 168)
    return pd.Series(values, index=idx)


def test_feature_model_predicts_horizon_with_intervals():
    s = _load_series(30 * 24)
    train = s.iloc[: 25 * 24]

    model = fm.fit_feature_model(train, n_estimators=8)
    yhat, pi = fm.predict_feature_model(model, train, h=6)

    assert len(yhat) == 6
    assert list(pi.columns) == ["lo", "hi"]
    assert (pi["hi"] >= pi["lo"]).all()
    assert yhat.index[0] == train.index[-1] + pd.Timedelta(hours=1)


def test_eval_feature_model_uses_common_backtesting_and_baseline(monkeypatch):
    s = _load_series()
    original_fit = fm.fit_feature_model

    def fast_fit(train, random_state=42):
        return original_fit(train, random_state=random_state, n_estimators=8)

    monkeypatch.setattr(fm, "fit_feature_model", fast_fit)

    summary, gain, details = fm.eval_feature_model(s, H=6, win_days=20, eval_days=2)

    assert not summary.empty
    assert summary["model"].iloc[0] == fm.MODEL_NAME
    assert "PI_coverage_pct" in summary.columns
    assert np.isfinite(gain)
    assert details["valid"].any()
