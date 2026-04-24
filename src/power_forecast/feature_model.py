import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .evaluation import baseline_gain_pct, score_forecast, summarize_scores, walk_forward_folds
from .features import FEATURE_COLUMNS, build_feature_row, build_supervised_frame


MODEL_NAME = "random_forest_lag_calendar"
MODEL_VERSION = "rf_lag_calendar_v1"


def _seasonal_naive(train: pd.Series, h: int, m: int = 168) -> pd.Series:
    last = train.iloc[-m:]
    vals = np.resize(last.values, h)
    idx = pd.date_range(train.index[-1] + pd.Timedelta(hours=1), periods=h, freq="h")
    return pd.Series(vals, index=idx)


def fit_feature_model(train: pd.Series, random_state: int = 42, n_estimators: int = 120) -> RandomForestRegressor:
    X, y = build_supervised_frame(train)
    if len(X) < 168:
        raise ValueError(f"Not enough training rows for feature model: {len(X)}")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=18,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_feature_model(
    model: RandomForestRegressor,
    train: pd.Series,
    h: int = 24,
    interval_q: tuple[float, float] = (0.05, 0.95),
) -> tuple[pd.Series, pd.DataFrame]:
    history = pd.to_numeric(train, errors="coerce").sort_index().asfreq("h").ffill()
    idx = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=h, freq="h")
    preds = []
    lo = []
    hi = []

    for ts in idx:
        Xrow = build_feature_row(history, ts)
        if Xrow.isna().any(axis=None):
            raise ValueError(f"Missing lag features for {ts}")

        pred = float(model.predict(Xrow)[0])
        preds.append(pred)

        tree_preds = np.asarray([tree.predict(Xrow.to_numpy())[0] for tree in model.estimators_], dtype=float)
        lo.append(float(np.quantile(tree_preds, interval_q[0])))
        hi.append(float(np.quantile(tree_preds, interval_q[1])))

        history.loc[ts] = pred

    yhat = pd.Series(preds, index=idx, name="yhat")
    pi = pd.DataFrame({"lo": lo, "hi": hi}, index=idx)
    return yhat, pi


def refit_predict_feature_model(s: pd.Series, H: int = 24, win_days: int = 90):
    train = s.sort_index().asfreq("h").tail(win_days * 24).ffill()
    model = fit_feature_model(train)
    return predict_feature_model(model, train, h=H)


def eval_feature_model(
    s: pd.Series,
    H: int = 24,
    win_days: int = 90,
    eval_days: int = 30,
    step_hours: int = 24,
    min_coverage: float = 0.8,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    rows = []
    min_points = max(2 * 168, win_days * 24 // 2)
    folds = walk_forward_folds(
        s,
        H=H,
        win_days=win_days,
        eval_days=eval_days,
        step_hours=step_hours,
        min_train_points=min_points,
    )

    for fold in folds:
        row_base = {
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "cutoff": fold.cutoff,
            "horizon": H,
        }
        try:
            model = fit_feature_model(fold.train)
            fc, pi = predict_feature_model(model, fold.train, h=len(fold.test))
            base = _seasonal_naive(fold.train, len(fc), m=168)
            score = score_forecast(
                fold.test,
                fc,
                baseline=base,
                insample=fold.train,
                mase_m=168,
                interval=pi,
                nominal_coverage=0.90,
                min_coverage=min_coverage,
            )
            rows.append({**row_base, **score, "error": ""})
        except Exception as exc:
            rows.append(
                {
                    **row_base,
                    "valid": False,
                    "points_compared": 0,
                    "expected_points": len(fold.test),
                    "coverage_pct": 0.0,
                    "MAE": np.nan,
                    "sMAPE": np.nan,
                    "MASE_168h": np.nan,
                    "MAE_base": np.nan,
                    "Gain": np.nan,
                    "PI_coverage_pct": np.nan,
                    "PI_mean_width_MW": np.nan,
                    "PI_calibration_error_pct": np.nan,
                    "error": str(exc),
                }
            )

    details = pd.DataFrame(rows)
    summary = summarize_scores(rows)
    if summary.empty:
        return summary, np.nan, details

    summary["model"] = MODEL_NAME
    summary["model_version"] = MODEL_VERSION
    summary["feature_count"] = len(FEATURE_COLUMNS)
    gain = baseline_gain_pct(float(summary["MAE"].iloc[0]), float(summary["MAE_base"].iloc[0]))
    return summary.round(3), gain, details


def eval_feature_model_windows(
    s: pd.Series,
    validation_windows: tuple[int, ...] = (7, 14, 30),
    H: int = 24,
    win_days: int = 90,
) -> pd.DataFrame:
    rows = []
    for eval_days in validation_windows:
        summary, gain, _ = eval_feature_model(s, H=H, win_days=win_days, eval_days=eval_days)
        if summary.empty:
            rows.append({"eval_days": eval_days, "model": MODEL_NAME, "Gain": np.nan})
            continue
        row = summary.iloc[0].to_dict()
        row["eval_days"] = eval_days
        row["Gain"] = gain
        rows.append(row)
    return pd.DataFrame(rows)
