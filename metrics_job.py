
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
                       spec_path=SPEC_PATH,
                       base_loc: Optional[pd.Series] = None) -> Path:

    df = pd.DataFrame({"yhat": yhat_loc})

    if base_loc is not None:
        df["yhat_snaive"] = base_loc

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
        if (now_loc - issue) >= pd.Timedelta(hours=24):
       # if (now_loc - issue) >= pd.Timedelta(minutes=5):  # zum testen
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


META_COLS_FLAT = [
    "k_exog", "train_window_days", "last_refit", "spec_sha256"
]

def _extract_meta_from_forecast(fc: pd.DataFrame, fpath: Path) -> dict:
    """
    Holt Meta-Infos aus der Forecast-CSV.

    """
    out = {}

    # 1) Direkt übernommene Felder
    for c in META_COLS_FLAT:
        if c in fc.columns:
            out[c] = fc[c].iloc[0]

    # 2) order / seasonal_order aus JSON-String parsen (wenn vorhanden)
    def _parse_list(val):
        if isinstance(val, str):
            try:
                return json.loads(val)               # "[1,1,1]" -> [1,1,1]
            except Exception:
                # zur Not aus "(1,1,1)" machen:
                try:
                    return json.loads(val.replace("(", "[").replace(")", "]"))
                except Exception:
                    return None
        return val

    if "order" in fc.columns:
        o = _parse_list(fc["order"].iloc[0])
        if isinstance(o, (list, tuple)):
            o = list(o) + [None, None, None]  # auffüllen
            out["order"]   = f"({o[0]},{o[1]},{o[2]})"
            out["order_p"] = o[0]; out["order_d"] = o[1]; out["order_q"] = o[2]

    if "seasonal_order" in fc.columns:
        so = _parse_list(fc["seasonal_order"].iloc[0])
        if isinstance(so, (list, tuple)):
            so = list(so) + [None, None, None, None]
            out["seasonal_order"] = f"({so[0]},{so[1]},{so[2]},{so[3]})"
            out["seasonal_P"] = so[0]; out["seasonal_D"] = so[1]
            out["seasonal_Q"] = so[2]; out["seasonal_m"] = so[3]

    return out



def evaluate_yesterday_and_save_today():
    now_loc = pd.Timestamp.now(tz="Europe/Berlin")
    fpath = _read_last_unscored_forecast(now_loc)
    if fpath is not None:
        fc = pd.read_csv(fpath, parse_dates=["ts"]).set_index("ts")
        fc.index = fc.index.tz_convert("Europe/Berlin")

        # Ist-Werte (lokal) laden
        s = load_smard_api(years=1)
        y_true = s.reindex(fc.index)

        # Inner Join / Intersection (keine NaNs)
        cols = {"y_true": y_true, "yhat": fc.get("yhat")}
        if "yhat_snaive" in fc.columns:
            cols["yhat_snaive"] = fc["yhat_snaive"]

        aligned = pd.concat(cols, axis=1).dropna()
        cov = len(aligned)

        if cov > 0:
            mae = mean_absolute_error(aligned.y_true, aligned.yhat)
            sm = smape(aligned.y_true, aligned.yhat)
            if "yhat_snaive" in aligned:
                mae_base = mean_absolute_error(aligned.y_true, aligned.yhat_snaive)
                gain = (mae_base - mae) / (mae_base + 1e-12) * 100
            else:
                mae_base = np.nan
                gain = np.nan
        else:
            mae = sm = mae_base = gain = np.nan

        row = {
            "scored_at": now_loc.strftime("%Y-%m-%d %H:%M"),
            "forecast_file": fpath.name,
            "forecast_issue": fpath.stem.replace("_", " "),
            "points_compared": cov,
            "MAE": round(float(mae), 3) if cov else np.nan,
            "sMAPE": round(float(sm), 3) if cov else np.nan,
            "MAE_base": round(float(mae_base), 3) if cov else np.nan,
            "Gain": round(float(gain), 1) if cov else np.nan,
        }

        row.update(_extract_meta_from_forecast(fc, fpath))

        _append_metrics_row(row)
        print(f"Scored {fpath.name} (coverage={cov})")

    # 2) ISSUE TODAY: generate today's forecast and save it for tomorrow's evaluation
    s = load_smard_api(years=1)
    spec = json.load(open(SPEC_PATH, "r", encoding="utf-8"))
    yhat_utc, pi_utc = forecast_from_params(s, H=24,win_days=spec["win_days"])
    yhat_loc, pi_loc = to_local(yhat_utc, pi_utc)

    #seas_naive forecast
    base_loc=s_naive(s,len(yhat_loc),m=168)

    path = _save_forecast_csv(yhat_loc, pi_loc, now_loc,base_loc=base_loc)
    print(f"Issued forecast and saved to {path}")

if __name__ == "__main__":
    evaluate_yesterday_and_save_today()
