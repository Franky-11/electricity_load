
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional
import json, hashlib

from forecast import forecast_from_params, to_local, smape,s_naive
from smard_data import load_smard_api
from sklearn.metrics import mean_absolute_error

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
FORECAST_DIR = ARTIFACTS_DIR / "forecasts"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)
METRICS_CSV = ARTIFACTS_DIR / "metrics.csv"
SPEC_PATH  = os.path.join(ARTIFACTS_DIR, "sarima_spec.json")

def _save_forecast_csv(yhat_loc: pd.Series,
                       pi_loc: Optional[pd.DataFrame],
                       issue_ts: pd.Timestamp,
                       spec_path=SPEC_PATH) -> Path:

    df = pd.DataFrame({"yhat": yhat_loc})
    if pi_loc is not None and not pi_loc.empty:
        for c in pi_loc.columns: df[c] = pi_loc[c]

    # Spec laden + stabile ID bilden
    meta = {}
    try:
        spec = json.load(open(spec_path, "r", encoding="utf-8"))
        meta = {
            "order": tuple(spec.get("order", [])),
            "seasonal_order": tuple(spec.get("seasonal_order", [])),
            "k_exog": spec.get("k_exog"),
            "train_window_days": spec.get("win_days"),
            "last_refit": spec.get("last_refit"),
        }
        spec_norm = json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")
        meta["spec_sha256"] = hashlib.sha256(spec_norm).hexdigest()[:12]

    except Exception:
        meta = {"spec_missing": True}

    # Meta als konstante Spalten anhängen
    for k, v in meta.items():
        if isinstance(v, (list, tuple, dict)):
            v = json.dumps(v, ensure_ascii=False)  # macht es zum Skalar-String
        df[k] = v

    fname = issue_ts.strftime("%Y-%m-%d_%H%M") + ".csv"
    path = FORECAST_DIR / fname
    df.to_csv(path, index_label="ts")

    # optional: Sidecar-JSON (gleiches Meta, bequemer fürs Parsen)
    try:
        (FORECAST_DIR / f"{path.stem}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
    return path


def _read_last_unscored_forecast(now_loc: pd.Timestamp) -> Optional[Path]:
    # Pick the most recent forecast file whose issue time is at least 24 hours in the past
    # and that is not yet present in metrics.csv (by filename).
    files = sorted(FORECAST_DIR.glob("*.csv"))
    if not files:
        return None
    df_m = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else pd.DataFrame(columns=["forecast_file"])
    already = set(df_m.get("forecast_file", pd.Series(dtype=str)).tolist())
    cand = []
    for f in files:
        # parse 'YYYY-MM-DD_HHMM.csv'
        stem = f.stem
        try:
            issue = pd.to_datetime(stem, format="%Y-%m-%d_%H%M").tz_localize("Europe/Berlin")
        except Exception:
            continue
        if f.name in already:
            continue
        #if (now_loc - issue) >= pd.Timedelta(hours=24):
        if (now_loc - issue) >= pd.Timedelta(minutes=5):  # zum testen
            cand.append((issue, f))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[0][1]

def _append_metrics_row(row: dict):
    df = pd.DataFrame([row])
    if METRICS_CSV.exists():
        old = pd.read_csv(METRICS_CSV)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(METRICS_CSV, index=False)

def evaluate_yesterday_and_save_today():
    now_loc = pd.Timestamp.now(tz="Europe/Berlin")

    # 1) EVALUATE: find last unscored forecast (>=24h old)
    fpath = _read_last_unscored_forecast(now_loc)
    if fpath is not None:
        fc = pd.read_csv(fpath, parse_dates=["ts"]).set_index("ts")
        fc.index = fc.index.tz_convert("Europe/Berlin")
        # columns: yhat, optional PI columns
        s = load_smard_api(years=1).tz_convert("Europe/Berlin")
        y_true = s.reindex(fc.index)
        mask = (~y_true.isna()) & (~fc["yhat"].isna())
        cov = int(mask.sum())
        mae = float(mean_absolute_error(y_true[mask], fc["yhat"][mask])) if cov > 0 else np.nan
        sm = float(smape(y_true[mask], fc["yhat"][mask])) if cov > 0 else np.nan
        base =s_naive(s, len(fc), m=168)
        mae_base= float(mean_absolute_error(y_true[mask], base[mask]))
        gain = float((mae_base- mae) / mae_base * 100) if cov > 0 else np.nan

        row = {
            "scored_at": now_loc.strftime("%Y-%m-%d %H:%M"),
            "forecast_file": fpath.name,
            "forecast_issue": fpath.stem.replace("_", " "),
            "points_compared": cov,
            "MAE": round(mae, 3) if cov > 0 else np.nan,
            "sMAPE": round(sm, 3) if cov > 0 else np.nan,
            "MAE_base": round(mae_base, 3) if cov > 0 else np.nan,
            "Gain": round(gain, 1) if cov > 0 else np.nan
        }

        meta_cols = ["order", "seasonal_order", "k_exog", "train_window_days", "last_refit", "spec_sha256"]
        row.update({c: (str(fc[c].iloc[0]) if c in fc.columns else None) for c in meta_cols})

        _append_metrics_row(row)
        print(f"Scored {fpath.name} with coverage={cov}")

    # 2) ISSUE TODAY: generate today's forecast and save it for tomorrow's evaluation
    s = load_smard_api(years=1)
    yhat_utc, pi_utc = forecast_from_params(s, H=24)
    yhat_loc, pi_loc = to_local(yhat_utc, pi_utc)
    path = _save_forecast_csv(yhat_loc, pi_loc, now_loc)
    print(f"Issued forecast and saved to {path}")

if __name__ == "__main__":
    evaluate_yesterday_and_save_today()
