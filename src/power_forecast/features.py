import numpy as np
import pandas as pd
from holidays import Germany

from .config import TIMEZONE


LAGS = (24, 48, 168)
ROLLING_WINDOWS = (24, 168)

CALENDAR_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
]

LAG_FEATURES = [f"lag_{lag}h" for lag in LAGS]
ROLLING_FEATURES = [f"rolling_mean_{w}h" for w in ROLLING_WINDOWS]
FEATURE_COLUMNS = CALENDAR_FEATURES + LAG_FEATURES + ROLLING_FEATURES


def _local_index(index: pd.DatetimeIndex, tz: str = TIMEZONE) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize("UTC").tz_convert(tz)
    return idx.tz_convert(tz)


def calendar_features(index: pd.DatetimeIndex, tz: str = TIMEZONE) -> pd.DataFrame:
    """Calendar features for UTC-naive model timestamps."""
    idx = pd.DatetimeIndex(index)
    idx_local = _local_index(idx, tz=tz)
    years = range(int(idx_local.year.min()), int(idx_local.year.max()) + 1)
    holidays_de = Germany(years=years)

    hour = idx_local.hour.to_numpy()
    dow = idx_local.dayofweek.to_numpy()
    month = idx_local.month.to_numpy()

    X = pd.DataFrame(index=idx)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    X["is_weekend"] = (idx_local.dayofweek >= 5).astype(int)
    X["is_holiday"] = pd.Series(idx_local.date, index=idx).map(lambda d: d in holidays_de).astype(int)
    return X[CALENDAR_FEATURES]


def build_supervised_frame(
    y: pd.Series,
    lags: tuple[int, ...] = LAGS,
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS,
    tz: str = TIMEZONE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a leakage-safe lag/calendar design matrix for one-step training."""
    s = pd.to_numeric(y, errors="coerce").sort_index().asfreq("h").ffill()
    X = calendar_features(s.index, tz=tz)

    for lag in lags:
        X[f"lag_{lag}h"] = s.shift(lag)
    for window in rolling_windows:
        X[f"rolling_mean_{window}h"] = s.shift(1).rolling(window, min_periods=window).mean()

    data = X.join(s.rename("target")).dropna()
    return data[FEATURE_COLUMNS], data["target"]


def build_feature_row(
    history: pd.Series,
    timestamp: pd.Timestamp,
    lags: tuple[int, ...] = LAGS,
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS,
    tz: str = TIMEZONE,
) -> pd.DataFrame:
    """Build one forecast row from history available strictly before timestamp."""
    ts = pd.Timestamp(timestamp)
    hist = pd.to_numeric(history, errors="coerce").sort_index().asfreq("h").ffill()
    row = calendar_features(pd.DatetimeIndex([ts]), tz=tz)

    for lag in lags:
        lag_ts = ts - pd.Timedelta(hours=lag)
        row[f"lag_{lag}h"] = hist.get(lag_ts, np.nan)

    hist_before = hist.loc[: ts - pd.Timedelta(hours=1)]
    for window in rolling_windows:
        vals = hist_before.tail(window)
        row[f"rolling_mean_{window}h"] = vals.mean() if len(vals) >= window else np.nan

    return row[FEATURE_COLUMNS]
