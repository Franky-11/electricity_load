from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import holidays
import streamlit as st



@st.cache_data()
def load_csv():
    PATH = "DE_load_actual_entsoe_transparency.csv"
    DATE_COL, Y_COL = "timestamp", "DE_load_actual_entsoe_transparency"
    FREQ = "h"
    # 1) Laden & sortieren
    df=pd.read_csv(PATH,parse_dates=[DATE_COL]).sort_values(DATE_COL)
    s=df.set_index(DATE_COL)[Y_COL]
    s.index = pd.to_datetime(s.index, errors="raise")  # aus generischem Index -> DatetimeIndex

    if s.index.tz is None:
        # -> lokale Berlin-Zeit ohne TZ: sauber lokalisieren (DST!)
        s.index = s.index.tz_localize("Europe/Berlin",
                                      ambiguous="infer",  # doppelte Stunde im Herbst
                                      nonexistent="shift_forward")  # fehlende Stunde im Frühjahr
    else:
        # -> bereits tz-aware (z.B. UTC+01:00): nach Berlin konvertieren
        s.index = s.index.tz_convert("Europe/Berlin")


    # 3) Feste Frequenz setzen (wichtig für Saisonmuster) und Nan entfernen (nur einer)
    s = s.asfreq(FREQ)
    s.dropna(inplace=True)
    return s


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
    return out.mean().round(3) if len(out) else out

def load_sarima_pred():
    df=pd.read_csv("sarima.csv",parse_dates=["timestamp"])
    return df.set_index("timestamp")


def format_thinspace(x: float) -> str:
    # 12 345 statt 12,345 – näher am Screenshot
    return f"{x:,.0f}".replace(",", " ")


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
