from dataclasses import dataclass

import pandas as pd

from config import TIMEZONE


@dataclass(frozen=True)
class DataQuality:
    last_local: pd.Timestamp
    lag_hours: float
    missing_last: int
    expected_last: int
    coverage_last_pct: float
    duplicates_total: int
    timezone: str
    window_days: int


def calculate_data_quality(s: pd.Series, tz: str = TIMEZONE, last: int = 90) -> DataQuality:
    idx = s.index
    if idx.tz is not None:
        idx_local = idx.tz_convert(tz)
        y = s.tz_convert("UTC").tz_localize(None).asfreq("h")
        last_local = idx_local.max()
        w_start = last_local - pd.Timedelta(days=last)
        y_last = y.loc[
            w_start.tz_convert("UTC").tz_localize(None): last_local.tz_convert("UTC").tz_localize(None)
        ]
    else:
        idx_local = idx.tz_localize(tz)
        y = s.asfreq("h")
        last_local = idx_local.max()
        w_start = last_local - pd.Timedelta(days=last)
        y_last = y.loc[w_start.tz_localize(None): last_local.tz_localize(None)]

    now_local = pd.Timestamp.now(tz)
    expected = last * 24
    missing_last = int(y_last.isna().sum())
    coverage_last = 100 * (1 - missing_last / max(expected, 1))

    return DataQuality(
        last_local=last_local,
        lag_hours=(now_local - last_local).total_seconds() / 3600,
        missing_last=missing_last,
        expected_last=expected,
        coverage_last_pct=coverage_last,
        duplicates_total=int(s.index.duplicated(keep="first").sum()),
        timezone=str(idx_local.tz),
        window_days=last,
    )
