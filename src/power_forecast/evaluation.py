from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestFold:
    cutoff: pd.Timestamp
    train: pd.Series
    test: pd.Series


def to_model_series(s: pd.Series) -> pd.Series:
    """Return a sorted, hourly, UTC-naive series for model evaluation."""
    y = s.copy()
    if y.index.tz is not None:
        y = y.tz_convert("UTC").tz_localize(None)
    y = y.sort_index()
    y = y[~y.index.duplicated(keep="last")]
    return y.asfreq("h")


def walk_forward_folds(
    s: pd.Series,
    H: int = 24,
    win_days: int = 90,
    eval_days: int = 30,
    step_hours: int = 24,
    min_train_points: int | None = None,
) -> list[BacktestFold]:
    y = to_model_series(s)
    folds = []
    cutoff = y.index.max() - pd.Timedelta(days=eval_days)

    while cutoff + pd.Timedelta(hours=H) <= y.index.max():
        train = y.loc[:cutoff].tail(win_days * 24).ffill()
        test = y.loc[cutoff + pd.Timedelta(hours=1): cutoff + pd.Timedelta(hours=H)]
        if len(test) and (min_train_points is None or len(train) >= min_train_points):
            folds.append(BacktestFold(cutoff=cutoff, train=train, test=test))
        cutoff += pd.Timedelta(hours=step_hours)

    return folds


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)).replace(0, np.finfo(float).eps)
    return float(200 * np.mean(np.abs(y_true - y_pred) / denom))


def mase(y_true: pd.Series, y_pred: pd.Series, insample: pd.Series, m: int = 1) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    insample = pd.to_numeric(insample, errors="coerce")
    if m == 1:
        scale = insample.diff().abs().dropna().mean()
    else:
        scale = (insample - insample.shift(m)).abs().dropna().mean()
    return float((np.abs(y_true - y_pred).mean()) / (float(scale) + 1e-12))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def score_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    baseline: pd.Series | None = None,
    insample: pd.Series | None = None,
    mase_m: int = 168,
    min_coverage: float = 0.8,
) -> dict:
    aligned = pd.concat({"y_true": y_true, "y_pred": y_pred}, axis=1)
    if baseline is not None:
        aligned["baseline"] = baseline
    aligned = aligned.dropna(subset=["y_true", "y_pred"])

    expected = len(y_pred)
    compared = len(aligned)
    coverage = compared / max(expected, 1)
    valid = coverage >= min_coverage and compared > 0

    row = {
        "points_compared": compared,
        "expected_points": expected,
        "coverage_pct": coverage * 100,
        "valid": valid,
        "MAE": np.nan,
        "sMAPE": np.nan,
        "MASE_168h": np.nan,
        "MAE_base": np.nan,
        "Gain": np.nan,
    }
    if not valid:
        return row

    yt = aligned["y_true"].astype(float)
    yp = aligned["y_pred"].astype(float)
    row["MAE"] = mae(yt, yp)
    row["sMAPE"] = smape(yt, yp)
    if insample is not None:
        row["MASE_168h"] = mase(yt, yp, insample, m=mase_m)

    if baseline is not None:
        base = aligned["baseline"].astype(float)
        row["MAE_base"] = mae(yt, base)
        row["Gain"] = (row["MAE_base"] - row["MAE"]) / (row["MAE_base"] + 1e-12) * 100

    return row


def summarize_scores(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    valid = df[df["valid"]].copy()
    if valid.empty:
        return pd.DataFrame()

    numeric = valid.select_dtypes(include=[np.number]).mean(numeric_only=True).to_frame().T
    numeric["folds"] = len(valid)
    return numeric.round(3)


def baseline_gain_pct(mae_model: float, mae_baseline: float) -> float:
    if pd.isna(mae_model) or pd.isna(mae_baseline):
        return np.nan
    return float((mae_baseline - mae_model) / (mae_baseline + 1e-12) * 100)
