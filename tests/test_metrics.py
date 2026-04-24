import numpy as np
import pandas as pd

from power_forecast.forecast import mase, s_naive, smape


def test_smape_matches_definition():
    y = pd.Series([100.0, 200.0])
    yhat = pd.Series([110.0, 180.0])

    expected = 200 * np.mean([10 / 210, 20 / 380])

    assert smape(y, yhat) == expected


def test_mase_uses_insample_naive_scale():
    y = pd.Series([12.0, 14.0])
    yhat = pd.Series([11.0, 15.0])
    insample = pd.Series([10.0, 12.0, 14.0, 16.0])

    assert mase(y, yhat, insample, m=1) == 0.5


def test_seasonal_naive_repeats_last_season_window():
    idx = pd.date_range("2026-01-01 00:00", periods=4, freq="h")
    tr = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx)

    fc = s_naive(tr, h=5, m=2)

    assert list(fc.values) == [30.0, 40.0, 30.0, 40.0, 30.0]
    assert fc.index[0] == pd.Timestamp("2026-01-01 04:00")
