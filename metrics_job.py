
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

def _pick_forecast_to_score(now_loc: pd.Timestamp) -> Optional[Path]:
    """Wähle die letzte Forecast-CSV für 'gestern' (Europe/Berlin).
    Fallback: wenn keine für gestern existiert, nimm die jüngste ungescorte vor gestern.
    """
    files = []
    for f in FORECAST_DIR.glob("*.csv"):
        try:
            issue = pd.to_datetime(f.stem, format="%Y-%m-%d_%H%M").tz_localize("Europe/Berlin")
            files.append((issue, f))
        except Exception:
            continue
    if not files:
        return None

    # bereits gescorte aus metrics.csv filtern
    df_m = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else pd.DataFrame(columns=["forecast_file"])
    already = set(df_m.get("forecast_file", pd.Series(dtype=str)).tolist())

    yday = (now_loc - pd.Timedelta(days=1)).date()

    # 1) Kandidaten für gestern, jüngste Uhrzeit zuerst
    cand_yday = sorted(
        [(iss, f) for iss, f in files if f.name not in already and iss.date() == yday],
        key=lambda x: x[0], reverse=True
    )
    if cand_yday:
        return cand_yday[0][1]

    # 2) Fallback: jüngste ungescorte vor gestern
    cand_older = sorted(
        [(iss, f) for iss, f in files if f.name not in already and iss.date() < yday],
        key=lambda x: x[0], reverse=True
    )
    return cand_older[0][1] if cand_older else None







def _append_metrics_row(row: dict):
    df = pd.DataFrame([row])
    if METRICS_CSV.exists():
        old = pd.read_csv(METRICS_CSV)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(METRICS_CSV, index=False)


META_COLS_FLAT = [
    "k_exog", "train_window_days", "last_refit", "spec_sha256"
]

def _parse_issue_from_artifact_name(fpath: Path) -> Optional[pd.Timestamp]:
    """Parst den Issue-Zeitstempel aus forecast csv/meta Dateinamen."""
    name = fpath.name
    if name.endswith(".csv"):
        raw = name[:-4]
    elif name.endswith(".meta.json"):
        raw = name[:-10]
    else:
        return None
    try:
        return pd.to_datetime(raw, format="%Y-%m-%d_%H%M").tz_localize("Europe/Berlin")
    except Exception:
        return None


def _cleanup_forecasts_keep_last_days(days_to_keep: int = 2) -> int:
    """Behält nur Artefakte der letzten N Kalendertage (csv + meta)."""
    if days_to_keep < 1:
        days_to_keep = 1

    tagged_files = []
    for fpath in FORECAST_DIR.iterdir():
        if not fpath.is_file():
            continue
        issue_ts = _parse_issue_from_artifact_name(fpath)
        if issue_ts is None:
            continue
        tagged_files.append((issue_ts.date(), fpath))

    if not tagged_files:
        return 0

    keep_dates = set(sorted({d for d, _ in tagged_files}, reverse=True)[:days_to_keep])
    deleted = 0

    for issue_date, fpath in tagged_files:
        if issue_date in keep_dates:
            continue
        try:
            fpath.unlink()
            deleted += 1
        except Exception as e:
            print(f"Cleanup failed for {fpath.name}: {e}")

    return deleted


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

def _read_forecast(fpath: Path) -> pd.DataFrame:
    # robustes Einlesen mit DST-Sicherheit
    df = pd.read_csv(fpath)

    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")  # immer erst nach UTC parsen
        if ts.isna().any():
            # Fallback: ohne UTC parsen und später lokalisieren (für alte Dateien)
            ts = pd.to_datetime(df["ts"], errors="coerce")
        df = df.drop(columns=["ts"])
        df.index = ts
        df.index.name = "ts"
    else:
        # sehr alte Dateien: Index ist ts
        df = pd.read_csv(fpath, index_col=0)
        df.index.name = "ts"
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    # Wenn noch tz-naiv -> als Berlin lokalisieren (DST sicher)
    if getattr(df.index, "tz", None) is None:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df.index = df.index.tz_localize(
            "Europe/Berlin",
            ambiguous="infer",           # Fall-Back nach der Zeitumstellung (doppelte Stunde)
            nonexistent="shift_forward", # Spring-Forward
        )
    else:
        # bereits tz-aware (UTC oder Offset) -> nach Berlin konvertieren
        df.index = df.index.tz_convert("Europe/Berlin")

    # optional: doppelte Timestamps (Sommer→Winter) konsolidieren
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    return df


def evaluate_yesterday_and_save_today():
    now_loc = pd.Timestamp.now(tz="Europe/Berlin")
    fpath = _pick_forecast_to_score(now_loc)
    if fpath is not None:
        fc = _read_forecast(fpath)

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

    deleted = _cleanup_forecasts_keep_last_days(days_to_keep=2)
    print(f"Cleanup complete: removed {deleted} old forecast artifacts (kept last 2 days).")

if __name__ == "__main__":
    evaluate_yesterday_and_save_today()
