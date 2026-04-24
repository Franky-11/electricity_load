import numpy as np
import pandas as pd

from power_forecast.features import FEATURE_COLUMNS, build_feature_row, build_supervised_frame


def test_build_supervised_frame_contains_calendar_lag_and_rolling_features():
    idx = pd.date_range("2026-01-01 00:00", periods=220, freq="h")
    y = pd.Series(50_000 + np.arange(len(idx), dtype=float), index=idx)

    X, target = build_supervised_frame(y)

    assert list(X.columns) == FEATURE_COLUMNS
    assert not X.isna().any(axis=None)
    assert len(X) == len(target)
    assert X.index.min() == idx[168]


def test_build_feature_row_uses_only_available_history():
    idx = pd.date_range("2026-01-01 00:00", periods=220, freq="h")
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    ts = idx[-1] + pd.Timedelta(hours=1)

    row = build_feature_row(y, ts)

    assert row.loc[ts, "lag_24h"] == y.loc[ts - pd.Timedelta(hours=24)]
    assert row.loc[ts, "lag_168h"] == y.loc[ts - pd.Timedelta(hours=168)]
    assert row.loc[ts, "rolling_mean_24h"] == y.tail(24).mean()
    assert not row.isna().any(axis=None)
