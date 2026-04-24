import numpy as np
import pandas as pd

from power_forecast.data_quality import calculate_data_quality


def test_calculate_data_quality_counts_recent_missing_values():
    idx = pd.date_range("2026-01-01 00:00", periods=24, freq="h", tz="Europe/Berlin")
    values = np.ones(24)
    values[5] = np.nan
    s = pd.Series(values, index=idx)

    quality = calculate_data_quality(s, last=1)

    assert quality.last_local == idx[-1]
    assert quality.expected_last == 24
    assert quality.missing_last == 1
    assert quality.coverage_last_pct == 100 * (1 - 1 / 24)
    assert quality.duplicates_total == 0
    assert quality.timezone == "Europe/Berlin"
