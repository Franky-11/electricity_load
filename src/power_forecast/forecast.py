import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.kalman_filter import MEMORY_NO_SMOOTHING

import gc

import holidays
import json
import os

from .config import (
    ARTIFACTS_DIR,
    PARAM_PATH,
    REPO_PARAM_FALLBACK,
    REPO_SPEC_FALLBACK,
    SPEC_PATH,
    VAL_PATH,
)
from .evaluation import (
    baseline_gain_pct,
    mase as calc_mase,
    score_forecast,
    smape as calc_smape,
    summarize_scores,
    walk_forward_folds,
)
from .feature_model import (
    MODEL_NAME as FEATURE_MODEL_NAME,
    MODEL_VERSION as FEATURE_MODEL_VERSION,
    eval_feature_model,
    eval_feature_model_windows,
    refit_predict_feature_model,
)
from .features import FEATURE_COLUMNS

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def resolve_params_spec():
    p = PARAM_PATH if os.path.exists(PARAM_PATH) else REPO_PARAM_FALLBACK
    s = SPEC_PATH  if os.path.exists(SPEC_PATH)  else REPO_SPEC_FALLBACK
    return p, s


def smape(y, yhat):
    return calc_smape(y, yhat)


# mase scaled auf (seasonal)_naive basleine
def mase(y: pd.Series, yhat: pd.Series, insample: pd.Series, m: int = 1) -> float:
    return calc_mase(y, yhat, insample, m=m)


def naive(tr, h):
    # 1) Forecast-Index: direkt NACH dem letzten Trainingszeitpunkt starten
    idx = pd.date_range(tr.index[-1] + pd.Timedelta(hours=1),
                        periods=h, freq="h")
    # 2) Persistenz: immer den letzten Trainingswert wiederholen
    return pd.Series(tr.iloc[-1], index=idx)


def s_naive(tr, h, m=24):
    # 1) letztes Saisonfenster holen (z. B. 24h bei Tagesmuster)
    last = tr.iloc[-m:]
    # 2) auf H Länge bringen: Muster wiederholen oder abschneiden
    vals = np.resize(last.values, h)  # (A)
    # 3) Forecast-Index wie oben
    idx = pd.date_range(tr.index[-1] + pd.Timedelta(hours=1),
                        periods=h, freq="h")
    return pd.Series(vals, index=idx)


def drift(tr, h):
    T = len(tr);
    slope = (tr.iloc[-1] - tr.iloc[0]) / (T - 1)
    idx = pd.date_range(tr.index[-1] + pd.Timedelta(hours=1), periods=h, freq="h")
    return pd.Series(tr.iloc[-1] + slope * np.arange(1, h + 1), index=idx)


def eval_baselines(s, H=24, m=24, win_days=90, eval_days=30):
    rows = []

    for fold in walk_forward_folds(s, H=H, win_days=win_days, eval_days=eval_days, step_hours=H):
        tr = fold.train
        te = fold.test
        for name, yhat in [("naive_last_hour", naive(tr, len(te))),
                           ("seasonal_naive_24h", s_naive(tr, len(te), m)),
                           ("seasonal_naive_168h", s_naive(tr, len(te), 168)),
                           ("drift", drift(tr, len(te)))]:
            row = score_forecast(te, yhat, insample=tr, mase_m=168)
            rows.append({"model": name,
                         "MAE (MW)": row["MAE"],
                         "sMAPE (%)": row["sMAPE"],
                         "MASE_168h": row["MASE_168h"]})
    if not rows:
        return pd.DataFrame(columns=["MAE (MW)", "sMAPE (%)", "MASE_168h"])
    return pd.DataFrame(rows).groupby("model").mean().round(3)




def _exog_utcnaive_to_local(idx_naive_utc: pd.DatetimeIndex) -> pd.DataFrame:
    idx_local = idx_naive_utc.tz_localize("UTC").tz_convert("Europe/Berlin")
    yrs = range(idx_local.min().year, idx_local.max().year + 1)
    hol = holidays.Germany(years=yrs)
    X = pd.DataFrame(index=idx_naive_utc)                 # <- Index bleibt UTC-naiv
    X["is_weekend"] = (idx_local.dayofweek >= 5).astype(int)
    X["is_hol"] = pd.Series(idx_local.date, index=idx_naive_utc).map(lambda d: d in hol).astype(int)
    return X



"""
def eval_sarimax_rolling90_fast(s: pd.Series, H=24, window_days=90, days_to_eval=30, step_hours=24):
    # 1) TZ-sicher nach UTC, dann TZ entfernen → keine DST-Duplikate
    s0 = s.copy()
    if s0.index.tz is not None:
        s0 = s0.tz_convert("UTC").tz_localize(None)
    s0 = s0.sort_index()
    s0 = s0[~s0.index.duplicated(keep="first")].asfreq("h")  # stündlicher Takt

    m = 168; rows = []
    t = s0.index.max() - pd.Timedelta(days=days_to_eval)      # nahe Serienende starten
    while t + pd.Timedelta(hours=H) <= s0.index.max():
        tr = s0.loc[:t].tail(24*window_days).ffill()          # rollierende 90 Tage
        te = s0.loc[t + pd.Timedelta(hours=1) : t + pd.Timedelta(hours=H)]
        if len(tr) < 2*m or len(te) == 0:                     # Guards
            t += pd.Timedelta(hours=step_hours); continue

        Xtr, Xte = _exog_utcnaive_to_local(tr.index), _exog_utcnaive_to_local(te.index)
        try:
            res = SARIMAX(tr, order=(1,0,0), seasonal_order=(0,1,0,m),
                          exog=Xtr, enforce_stationarity=False, enforce_invertibility=False
                         ).fit(disp=False, maxiter=200)
            fc = res.get_forecast(steps=len(te), exog=Xte).predicted_mean
            fc = pd.Series(np.asarray(fc), index=te.index)    # identische Labels erzwingen
            rows.append({"MAE": mean_absolute_error(te, fc),
                         "sMAPE": smape(te, fc),
                         "MASE168": mase(te, fc, tr, m=168)})
        except Exception as e:
            # zum Debuggen: print("Fold skipped:", e)
            pass
        t += pd.Timedelta(hours=step_hours)

    out = pd.DataFrame(rows)
    return out.mean().round(3) if len(out) else out

"""

def eval_sarimax_rolling90_fast(s: pd.Series, H=24, window_days=90, days_to_eval=30, step_hours=24):
    m = 168
    rows = []
    folds = walk_forward_folds(
        s,
        H=H,
        win_days=window_days,
        eval_days=days_to_eval,
        step_hours=step_hours,
        min_train_points=2 * m,
    )
    for fold in folds:
        tr = fold.train
        te = fold.test
        # 2) Exogene (float + feste Reihenfolge)
        Xtr = _exog_utcnaive_to_local(tr.index).astype("float64")
        Xte = _exog_utcnaive_to_local(te.index).astype("float64")
        cols = ["is_weekend", "is_hol"]
        if all(c in Xtr.columns for c in cols):
            Xtr, Xte = Xtr[cols], Xte[cols]

        try:
            # 3) Leichtgewichtig fitten (nur Parameter), dann filtern mit Memory-Schutz
            mod = SARIMAX(tr, order=(1,0,0), seasonal_order=(0,1,0,m),
                          exog=Xtr, enforce_stationarity=False, enforce_invertibility=False)
            try:
                params_hat = mod.fit(method="lbfgs", maxiter=150, disp=False,
                                     return_params=True, low_memory=True)
            except TypeError:
                params_hat = mod.fit(method="lbfgs", maxiter=150, disp=False, return_params=True)

            res = mod.filter(params_hat, conserve_memory=MEMORY_NO_SMOOTHING)

            fc_obj = res.get_forecast(steps=len(te), exog=Xte)
            fc = pd.Series(np.asarray(fc_obj.predicted_mean), index=te.index)
            pi = fc_obj.conf_int(alpha=0.05)
            pi.columns = ["lo", "hi"]
            pi.index = te.index
            base = s_naive(tr, len(fc), m=m)

            row = score_forecast(te, fc, baseline=base, insample=tr, mase_m=168, interval=pi, nominal_coverage=0.95)
            row["cutoff"] = fold.cutoff
            row["model"] = "sarima_exog"
            row["model_version"] = "sarima_exog_v1"
            row["error"] = ""
            rows.append(row)

            # 5) Speicher freigeben
            del fc_obj, res, mod, Xtr, Xte
            if len(rows) % 2 == 0:
                gc.collect()

        except Exception as exc:
            rows.append({
                "cutoff": fold.cutoff,
                "model": "sarima_exog",
                "model_version": "sarima_exog_v1",
                "valid": False,
                "points_compared": 0,
                "expected_points": len(te),
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
            })

    summary = summarize_scores(rows)
    if not summary.empty:
        gain = baseline_gain_pct(float(summary["MAE"].iloc[0]), float(summary["MAE_base"].iloc[0]))
        return summary.round(3), gain
    return summary, np.nan



def save_validation_json(df,gain, meta, path=VAL_PATH):
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    obj = {"meta": meta, "data": df.round(3).to_dict(orient="records"),"gain":gain}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_validation_json(path=VAL_PATH):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return pd.DataFrame(obj.get("data", [])), obj.get("meta", {}),obj.get("gain",5)



def load_sarima_pred():
    df=pd.read_csv("sarima.csv",parse_dates=["timestamp"])
    return df.set_index("timestamp")


"""
def refit_predict_pi(s, H=24, win_days=90, m=168, alpha=0.05):
    y = s.asfreq("h")
    if y.index.tz is not None: y = y.tz_convert("UTC").tz_localize(None)
    tr = y.tail(24*win_days).ffill()
    idx_f = pd.date_range(y.index[-1]+pd.Timedelta(hours=1), periods=H, freq="h")

    res = SARIMAX(tr, order=(1,0,0), seasonal_order=(0,1,0,m),
                  exog=_exog_utcnaive_to_local(tr.index), enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False, maxiter=200)

    fc = res.get_forecast(H, exog=_exog_utcnaive_to_local(idx_f))
    yhat = pd.Series(np.asarray(fc.predicted_mean), index=idx_f)
    pi = fc.conf_int(alpha=alpha); pi.columns = ["lo","hi"]; pi.index = idx_f
    save_model_light(res, win_days=win_days, params_path=PARAM_PATH, spec_path=SPEC_PATH)

    return yhat, pi, res

"""


def refit_predict_pi(s, H=24, win_days=90, m=168, alpha=0.05,
                     params_path=PARAM_PATH, spec_path=SPEC_PATH):
    # 1) Daten vorbereiten (UTC-naiv)
    y = s.asfreq("h")
    if y.index.tz is not None:
        y = y.tz_convert("UTC").tz_localize(None)
    tr = y.tail(24*win_days).ffill()
    idx_f = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), periods=H, freq="h")

    # 2) Exogene (float + feste Spaltenreihenfolge)
    Xtr = _exog_utcnaive_to_local(tr.index).astype("float64")
    Xf  = _exog_utcnaive_to_local(idx_f).astype("float64")
    cols = ["is_weekend", "is_hol"]
    if all(c in Xtr.columns for c in cols):
        Xtr, Xf = Xtr[cols], Xf[cols]

    # 3) Spezifikation + Warmstart (falls Snapshot da)
    order, seas = (1,0,0), (0,1,0,m)
    start_params = None
    try:
        spec = json.load(open(spec_path, "r", encoding="utf-8"))
        order = tuple(spec.get("order", order))
        seas  = tuple(spec.get("seasonal_order", seas))
        start_params = np.load(params_path)["params"]
    except Exception:
        pass

    mod = SARIMAX(tr, order=order, seasonal_order=seas, exog=Xtr,
                  enforce_stationarity=False, enforce_invertibility=False)

    # 4) PARAMETER schätzen (leichtgewichtig), dann FILTERn (speicherschonend)
    try:
        params_hat = mod.fit(start_params=start_params, method="lbfgs",
                             maxiter=150, disp=False, return_params=True, low_memory=True)
    except TypeError:
        params_hat = mod.fit(start_params=start_params, method="lbfgs",
                             maxiter=150, disp=False, return_params=True)

    res = mod.filter(params_hat, conserve_memory=MEMORY_NO_SMOOTHING)

    # 5) Forecast + PI
    fc = res.get_forecast(H, exog=Xf)
    yhat = pd.Series(np.asarray(fc.predicted_mean), index=idx_f, name="yhat")
    pi = fc.conf_int(alpha=alpha)
    pi.columns = ["lo", "hi"]; pi.index = idx_f

    # 6) Snapshot speichern (klein) + Aufräumen
    save_model_light(res,win_days=win_days, params_path=params_path, spec_path=spec_path)
    del fc, res, mod, Xtr, Xf; gc.collect()

    return yhat, pi




def forecast_from_params(s, H=24,win_days=90, params_path=PARAM_PATH, spec_path=SPEC_PATH):
    # a) Aktuelle letzten 90 Tage vorbereiten (UTC-naiv)
    y = s.asfreq("h")
    if y.index.tz is not None: y = y.tz_convert("UTC").tz_localize(None)
    tr = y.tail(24*win_days).ffill()
    idx_f = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), periods=H, freq="h")

    # b) Exogene exakt wie im Refit
    Xtr = _exog_utcnaive_to_local(tr.index)
    Xf  = _exog_utcnaive_to_local(idx_f)

    # c) Spezifikation + Parameter laden
    spec   = json.load(open(spec_path))
    params = np.load(params_path)["params"]

    # d) Modell rekonstruieren & nur filtern (keine Optimierung)
    mod = SARIMAX(tr, order=tuple(spec["order"]),
                  seasonal_order=tuple(spec["seasonal_order"]),
                  exog=Xtr, enforce_stationarity=False, enforce_invertibility=False)
    res0 = mod.filter(params)  # schnell

    fc = res0.get_forecast(H, exog=Xf)
    yhat = pd.Series(np.asarray(fc.predicted_mean), index=idx_f, name="yhat")
    pi = fc.conf_int(); pi.columns = ["lo","hi"]; pi.index = idx_f
    return yhat, pi



def to_local(yhat: pd.Series, pi: pd.DataFrame):
    yhat_loc = yhat.copy()
    if yhat_loc.index.tz is None:
        yhat_loc.index = yhat_loc.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    else:
        yhat_loc.index = yhat_loc.index.tz_convert("Europe/Berlin")
    pi_loc = pi.copy(); pi_loc.index = yhat_loc.index
    return yhat_loc, pi_loc


def save_model_light(res,win_days, params_path=PARAM_PATH, spec_path=SPEC_PATH):
    spec = {
        "model_name":    "sarima_exog",
        "model_version": "sarima_exog_v1",
        "order":         list(res.model.order),
        "seasonal_order":list(res.model.seasonal_order),
        "k_exog":        int(getattr(res.model, "k_exog", 0)),
        "feature_list":  ["is_weekend", "is_hol"],
        "target":        "load_mw",
        "last_refit":    pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "win_days":      win_days
    }
    np.savez_compressed(params_path, params=np.asarray(res.params))
    json.dump(spec, open(spec_path, "w"))


def backtest_current_model(s, H=24, eval_days=14, win_days=60, m=168,
                           session_res=None, spec_path="sarima_spec.json", params_path="sarima_params.npz"):
    # 2) Params + Spec holen (Session-Res bevorzugt)
    if session_res is not None:
        spec = {"order": list(session_res.model.order), "seasonal_order": list(session_res.model.seasonal_order)}
        params = np.asarray(session_res.params)
    else:
        spec = json.load(open(spec_path)); params = np.load(params_path)["params"]

    rows = []
    folds = walk_forward_folds(s, H=H, win_days=win_days, eval_days=eval_days, step_hours=H, min_train_points=2 * m)
    for fold in folds:
        tr = fold.train
        te = fold.test

        # --- Exogene float + feste Spaltenreihenfolge (robust) ---
        Xtr = _exog_utcnaive_to_local(tr.index).astype("float64")
        Xte = _exog_utcnaive_to_local(te.index).astype("float64")
        cols = ["is_weekend", "is_hol"]
        if all(c in Xtr.columns for c in cols):
            Xtr = Xtr[cols]; Xte = Xte[cols]

        # --- Modell & FILTER mit Memory-Schonung ---
        mod = SARIMAX(tr,
                      order=tuple(spec["order"]),
                      seasonal_order=tuple(spec["seasonal_order"]),
                      exog=Xtr,
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.filter(params, conserve_memory=MEMORY_NO_SMOOTHING)  # <= HIER

        # --- Forecast ---
        fc_obj = res.get_forecast(H, exog=Xte)
        fc = pd.Series(np.asarray(fc_obj.predicted_mean), index=te.index)
        pi = fc_obj.conf_int(alpha=0.05)
        pi.columns = ["lo", "hi"]
        pi.index = te.index
        base = s_naive(tr, len(fc), m=m)

        # --- robustes Scoring ---
        row = score_forecast(te, fc, baseline=base, insample=tr, mase_m=m, interval=pi, nominal_coverage=0.95)
        row["cutoff"] = fold.cutoff
        rows.append(row)

        # Speicher freigeben (wichtig bei Compose/Docker)
        del fc_obj, res, mod, Xtr, Xte
        if len(rows) % 2 == 0:
            gc.collect()

    summary = summarize_scores(rows)
    if summary.empty:
        return pd.Series({"MAE": np.nan, "sMAPE": np.nan}), np.nan
    gain = baseline_gain_pct(float(summary["MAE"].iloc[0]), float(summary["MAE_base"].iloc[0]))
    return summary.iloc[0][["MAE", "sMAPE"]].round(3), gain



# --- Model Card-------------#

def model_card_meta(kpis: dict | None = None, gain: float | None = None,
                    spec_path: str = SPEC_PATH, val_path: str = VAL_PATH) -> dict:
    meta = {}
    # Spezifikation laden
    try:
        spec = json.load(open(spec_path, "r", encoding="utf-8"))
        meta.update({
            "model_name": spec.get("model_name", "sarima_exog"),
            "model_version": spec.get("model_version", "sarima_exog_v1"),
            "order": tuple(spec.get("order", [])),
            "seasonal_order": tuple(spec.get("seasonal_order", [])),
            "k_exog": spec.get("k_exog"),
            "feature_list": spec.get("feature_list", ["is_weekend", "is_hol"]),
            "target": spec.get("target", "load_mw"),
            "train_window_days": spec.get("win_days"),
            "last_refit": spec.get("last_refit")
        })
    except Exception:
        pass
    # Letzte Validierung (falls vorhanden)
    try:
        df, vmeta, g = load_validation_json(val_path)
        if kpis is None and isinstance(df, pd.DataFrame) and not df.empty:
            row = df.iloc[0]
            kpis = {"MAE": float(row.get("MAE", np.nan)),
                    "sMAPE": float(row.get("sMAPE", np.nan)),
                    "PI_coverage_pct": float(row.get("PI_coverage_pct", np.nan)),
                    "PI_mean_width_MW": float(row.get("PI_mean_width_MW", np.nan))}
        if gain is None:
            gain = g
        meta["validated_at"] = vmeta.get("validated_at")
        meta["eval_days"] = vmeta.get("eval_days")
        meta["H"] = vmeta.get("H", 24)
    except Exception:
        pass
    # Runtime-Umgebung
    import sys, platform
    try:
        import statsmodels as sm
        smv = sm.__version__
    except Exception:
        smv = "?"
    meta["env"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "statsmodels": smv,
        "platform": platform.system()
    }
    if kpis is not None: meta["kpis"] = kpis
    if gain is not None: meta["gain"] = gain
    return meta


def model_card_markdown(meta: dict) -> str:
    def fmt(v):
        if v is None: return "—"
        if isinstance(v, float) and np.isnan(v): return "—"
        return str(v)
    k = meta.get("kpis", {})
    g = meta.get("gain", None)
    gtxt = "—" if g is None or (isinstance(g, float) and np.isnan(g)) else f"{g:.1f}%"
    lines = [
        "# Model Card – SARIMA",
        "## Spec",
        f"- model_name: `{meta.get('model_name', '—')}`",
        f"- model_version: `{meta.get('model_version', '—')}`",
        f"- target: `{meta.get('target', '—')}`",
        f"- order: `{meta.get('order', '—')}`",
        f"- seasonal_order: `{meta.get('seasonal_order', '—')}`",
        f"- k_exog: {fmt(meta.get('k_exog'))}",
        f"- features: `{', '.join(meta.get('feature_list', []) or [])}`",
        f"- train_window_days: {fmt(meta.get('train_window_days'))}",
        f"- last_refit: {fmt(meta.get('last_refit'))}",
        "## Candidate Model",
        f"- model_name: `{FEATURE_MODEL_NAME}`",
        f"- model_version: `{FEATURE_MODEL_VERSION}`",
        f"- features: `{', '.join(FEATURE_COLUMNS)}`",
        "## Validation (letzte)",
        f"- validated_at: {fmt(meta.get('validated_at'))}",
        f"- horizon H: {fmt(meta.get('H'))}",
        f"- eval_days: {fmt(meta.get('eval_days'))}",
        f"- MAE: {fmt(k.get('MAE'))}",
        f"- sMAPE: {fmt(k.get('sMAPE'))}",
        f"- PI coverage: {fmt(k.get('PI_coverage_pct'))}",
        f"- PI mean width MW: {fmt(k.get('PI_mean_width_MW'))}",
        f"- Vorteil ggü. s-Naive(168): {gtxt}",
        "## Environment",
        f"- python: {meta['env'].get('python', '—')}",
        f"- numpy: {meta['env'].get('numpy', '—')}",
        f"- pandas: {meta['env'].get('pandas', '—')}",
        f"- statsmodels: {meta['env'].get('statsmodels', '—')}",
        f"- platform: {meta['env'].get('platform', '—')}",
        "## Files",
        f"- params: `{PARAM_PATH}`",
        f"- spec: `{SPEC_PATH}`",
        f"- validation: `{VAL_PATH}`",
    ]
    return "\n".join(lines)
