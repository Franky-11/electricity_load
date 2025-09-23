from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.statespace.kalman_filter import (
    MEMORY_NO_FORECAST_MEAN, MEMORY_NO_FORECAST_COV,
    MEMORY_NO_PREDICTED, MEMORY_NO_FILTERED, MEMORY_NO_SMOOTHING)

import holidays
import streamlit as st
import json


def smape(y, yhat):
    d = (np.abs(y) + np.abs(yhat)).replace(0, np.finfo(float).eps)
    return 200 * np.mean(np.abs(y - yhat) / d)


# mase scaled auf (seasonal)_naive basleine
def mase(y: pd.Series, yhat: pd.Series, insample: pd.Series, m: int = 1) -> float:
    # 1) Align & cast
    y, yhat = y.align(yhat, join="inner")
    y = pd.to_numeric(y, errors="coerce");
    yhat = pd.to_numeric(yhat, errors="coerce")
    insample = pd.to_numeric(insample, errors="coerce")

    # 2) Skalenfaktor aus TRAIN
    if m == 1:
        scale = insample.diff().abs().dropna().mean()
    else:
        scale = (insample - insample.shift(m)).abs().dropna().mean()

    # 3) MASE
    return (np.abs(y - yhat).mean()) / (float(scale) + 1e-12)


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
    t = s.index.max() - pd.Timedelta(days=eval_days);
    rows = []

    while t + pd.Timedelta(hours=H) <= s.index.max():
        tr = s.loc[:t].tail(win_days * 24).ffill()  # nur TRAIN füllen
        te = s.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=H)]
        for name, yhat in [("naive_letzte Stunde", naive(tr, len(te))),
                           ("snaive_letzten 24 Stunden", s_naive(tr, len(te), m)),
                           ("snaive_letzten 168 Stunden", s_naive(tr, len(te), 168)),
                           ("drift", drift(tr, len(te)))]:
            rows.append({"model": name,
                         "MAE (MW)": mean_absolute_error(te, yhat),
                         "sMAPE (%)": smape(te, yhat),
                       #  "MASE1": mase(te, yhat, tr, m=1),
                       #  "MASE24": mase(te, yhat, tr, m=24),
                         "MASE_168h": mase(te, yhat, tr, m=168)})
        t += pd.Timedelta(hours=H)
    return pd.DataFrame(rows).groupby("model").mean().round(3)




def _exog_utcnaive_to_local(idx_naive_utc: pd.DatetimeIndex) -> pd.DataFrame:
    idx_local = idx_naive_utc.tz_localize("UTC").tz_convert("Europe/Berlin")
    yrs = range(idx_local.min().year, idx_local.max().year + 1)
    hol = holidays.Germany(years=yrs)
    X = pd.DataFrame(index=idx_naive_utc)                 # <- Index bleibt UTC-naiv
    X["is_weekend"] = (idx_local.dayofweek >= 5).astype(int)
    X["is_hol"] = pd.Series(idx_local.date, index=idx_naive_utc).map(lambda d: d in hol).astype(int)
    return X

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
                         "sMAPE (%)": smape(te, fc),
                         "MASE_168": mase(te, fc, tr, m=168)})
        except Exception as e:
            # zum Debuggen: print("Fold skipped:", e)
            pass
        t += pd.Timedelta(hours=step_hours)

    out = pd.DataFrame(rows)
    summary = (out.mean(numeric_only=True).to_frame().T if len(out) else out)
    return summary.round(3)


VAL_PATH = "artifacts/val_sarima_latest.json"
def save_validation_json(df: pd.DataFrame, meta: dict, path: str = VAL_PATH):
    obj = {"meta": meta, "data": df.round(3).to_dict(orient="records")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_validation_json(path: str = VAL_PATH):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    df = pd.DataFrame(obj.get("data", []))
    meta = obj.get("meta", {})
    return df, meta


def load_sarima_pred():
    df=pd.read_csv("sarima.csv",parse_dates=["timestamp"])
    return df.set_index("timestamp")


def format_thinspace(x: float) -> str:
    # 12 345 statt 12,345
    return f"{x:,.2f}".replace(",", " ")


def kpi_card(title: str, value: float, unit: str = "", icon: str = "⚡", footnote: str | None = None):
    cls = "pos" if value > 0 else "neg" if value < 0 else "neu"
    sign_icon = "🔺" if value > 0 else "🔻" if value < 0 else "⏸️"
    icon = icon or sign_icon  # optional eigenes Icon überschreibt das
    val = format_thinspace(value)
    st.markdown(f"""
    <div class="kpi {cls}">
      <div class="kpi-head"><span class="kpi-ic">{icon}</span>{title}</div>
      <div class="kpi-val">{val}<span class="kpi-unit"> {unit}</span></div>
      {f'<div class="kpi-foot">{footnote}</div>' if footnote else ''}
    </div>
    """, unsafe_allow_html=True)



def refit_predict_pi(s, H=24, win_days=60, m=168, alpha=0.05):
    y = s.asfreq("H")
    if y.index.tz is not None: y = y.tz_convert("UTC").tz_localize(None)
    tr = y.tail(24*win_days).ffill()
    idx_f = pd.date_range(y.index[-1]+pd.Timedelta(hours=1), periods=H, freq="h")

    def exog(idx):
        loc = idx.tz_localize("UTC").tz_convert("Europe/Berlin")
        yrs = range(loc.min().year, loc.max().year+1); hol = holidays.Germany(years=yrs)
        X = pd.DataFrame(index=idx)
        X["is_weekend"] = (loc.dayofweek>=5).astype(int)
        X["is_hol"] = pd.Series(loc.date, index=idx).map(lambda d: d in hol).astype(int)
        return X

    res = SARIMAX(tr, order=(1,0,0), seasonal_order=(0,1,0,m),
                  exog=exog(tr.index), enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False, maxiter=200)

    fc = res.get_forecast(H, exog=exog(idx_f))
    yhat = pd.Series(np.asarray(fc.predicted_mean), index=idx_f)
    pi = fc.conf_int(alpha=alpha); pi.columns = ["lo","hi"]; pi.index = idx_f
    save_model_light(res,win_days=win_days, params_path="sarima_params.npz", spec_path="sarima_spec.json")


    return yhat, pi, res


def forecast_from_params(s, H=24, params_path="sarima_params.npz", spec_path="sarima_spec.json"):
    # a) Aktuelle letzten 90 Tage vorbereiten (UTC-naiv)
    y = s.asfreq("h")
    if y.index.tz is not None: y = y.tz_convert("UTC").tz_localize(None)
    tr = y.tail(24*90).ffill()
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
    yhat_loc = yhat.tz_localize("UTC").tz_convert("Europe/Berlin")
    pi_loc = pi.copy(); pi_loc.index = yhat_loc.index
    return yhat_loc, pi_loc


def save_model_light(res,win_days, params_path="sarima_params.npz", spec_path="sarima_spec.json"):
    spec = {
        "order":         list(res.model.order),
        "seasonal_order":list(res.model.seasonal_order),
        "k_exog":        int(getattr(res.model, "k_exog", 0)),
        "last_refit":    pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "win_days":      win_days
    }
    np.savez_compressed(params_path, params=np.asarray(res.params))
    json.dump(spec, open(spec_path, "w"))


def backtest_current_model(s, H=24, eval_days=14, win_days=60, m=168,
                           session_res=None, spec_path="sarima_spec.json", params_path="sarima_params.npz"):
    # 1) Serie UTC-naiv & stündlich
    y = s.asfreq("h");  y = y.tz_convert("UTC").tz_localize(None) if y.index.tz is not None else y
    y = y.sort_index().asfreq("h"); rows=[]
    # 2) Params + Spec holen (Session-Res bevorzugt)
    if session_res is not None:
        spec = {"order": list(session_res.model.order), "seasonal_order": list(session_res.model.seasonal_order)}
        params = np.asarray(session_res.params)
    else:
        spec = json.load(open(spec_path)); params = np.load(params_path)["params"]
    # 3) Walk-forward
    t = y.index.max() - pd.Timedelta(days=eval_days)


    while t + pd.Timedelta(hours=H) <= y.index.max():
        tr = y.loc[:t].tail(win_days*24).ffill(); te_idx = pd.date_range(t+pd.Timedelta(hours=1), periods=H, freq="h")
        if len(tr)<2*m:
            t+=pd.Timedelta(hours=H)
            continue

        Xtr, Xte = _exog_utcnaive_to_local(tr.index), _exog_utcnaive_to_local(te_idx)

        mod = SARIMAX(tr,
                      order=tuple(spec["order"]),
                      seasonal_order=tuple(spec["seasonal_order"]),
                      exog=Xtr,
                      enforce_stationarity=False, enforce_invertibility=False)


        res = mod.filter(params)
        fc = pd.Series(np.asarray(res.get_forecast(H, exog=Xte).predicted_mean), index=te_idx)
        base = s_naive(tr, len(fc), m=m)


        # --- robustes Scoring ohne NaN-Crash ---
        y_true = y.reindex(fc.index).astype(float)  # 1) Ziel an fc-Index ausrichten
        base = s_naive(tr, len(fc), m=m).reindex(fc.index).astype(float)

        mask = (~y_true.isna()) & (~fc.isna())  # 2) Nur vollständige Paare
        cov = int(mask.sum())
        if cov < int(0.8 * len(fc)):  # 3) Zu wenig Abdeckung? -> Fold skippen
            t += pd.Timedelta(hours=H);
            continue

        yt, yp = y_true[mask], fc[mask]

        rows.append({
            "MAE": mean_absolute_error(yt, yp),
            "sMAPE": smape(yt, yp),
            "MAE_base": mean_absolute_error(yt, base[mask])
        })

        t += pd.Timedelta(hours=H)
    df = pd.DataFrame(rows)

    return (df[["MAE","sMAPE"]].mean().round(3),
            float((df["MAE_base"].mean() - df["MAE"].mean())/df["MAE_base"].mean()*100) if len(df) else np.nan)
