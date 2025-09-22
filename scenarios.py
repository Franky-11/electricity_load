
import numpy as np, pandas as pd

# A) Feiertag/Wochenende: Multiplikator (z.B. Feiertage 0.9, Wochenende 0.95)
def mult_holiday_weekend(y: pd.Series, holidays: set, hol_mult=0.9, weekend_mult=0.95):
    y = y.copy()
    idx = y.index.tz_convert("Europe/Berlin")
    is_hol = pd.Index(idx.date).isin(holidays)
    is_we = pd.Series(idx.weekday >= 5, index=y.index)
    fac = pd.Series(1.0, index=y.index)
    fac[is_we.values] *= weekend_mult
    fac[is_hol] *= hol_mult
    return y * fac

# B) Lastverschiebung: verschiebe Anteil frac von src_hours -> dst_hours (Energie erhalten)
def shift_load(y: pd.Series, frac=0.1, src_hours=range(18,22), dst_hours=range(2,6)):
    y = y.copy()
    loc = y.index.tz_convert("Europe/Berlin")
    df = pd.DataFrame({"y": y.values, "h": loc.hour, "d": loc.date}, index=y.index)
    src = df["h"].isin(src_hours); dst = df["h"].isin(dst_hours)
    take = df.loc[src, "y"] * frac; df.loc[src, "y"] -= take
    add_per_day = take.groupby(df["d"]).sum() / max(1, df.loc[dst, "h"].groupby(df.loc[dst,"d"]).count().mean())
    df.loc[dst, "y"] += df.loc[dst, "d"].map(add_per_day).fillna(0.0)
    return pd.Series(df["y"].values, index=y.index)

# C) Temperatur-Sensitivität (ohne Wetterdaten): lineare %-Änderung je °C-Abweichung
# Beispiel: delta_c = +5°C, k_perc_per_c = +1.5%/°C -> Faktor = 1 + 0.015*5
def temp_adjust(y: pd.Series, delta_c=0.0, k_perc_per_c=0.015, mode="multiplicative"):
    if mode == "multiplicative":
        return y * (1.0 + k_perc_per_c * float(delta_c))
    else:
        return y + y.mean() * (k_perc_per_c * float(delta_c))

# D) Synthetische PV (mittags-Glocke) in MW -> Netto-Last = y - pv
def synthetic_pv(y_index: pd.DatetimeIndex, capacity_mw=1000.0):
    idx = y_index.tz_convert("Europe/Berlin")
    h = idx.hour.values
    # Glocke um 13 Uhr, Breite ~4h; nachts = 0
    w = np.exp(-0.5 * ((h - 13) / 2.5) ** 2)
    w[(h < 6) | (h > 20)] = 0.0
    pv = capacity_mw * (w / (w.max() if w.max()>0 else 1))
    return pd.Series(pv, index=y_index)

# E) Synthetischer Wind (flach über Tag), Skalierung in MW
def synthetic_wind(y_index: pd.DatetimeIndex, capacity_mw=1000.0, capacity_factor=0.35):
    cf_hourly = np.full(len(y_index), capacity_factor)
    # leichte Nacht-Überhöhung
    hours = y_index.tz_convert("Europe/Berlin").hour.values
    cf_hourly += (hours < 6) * 0.05
    return pd.Series(capacity_mw * np.clip(cf_hourly, 0, 1), index=y_index)

# F) Effizienz-/Spar-Trend: ab start_date exponentiell pro Jahr (z.B. -1%/a)
def efficiency_trend(y: pd.Series, start_date=None, annual_rate=-0.01):
    y = y.copy()
    idx = y.index.tz_convert("Europe/Berlin")
    if start_date is None: start_date = idx.min()
    start_date = pd.to_datetime(start_date).tz_localize(idx.tz)
    t = (idx - start_date) / pd.Timedelta(days=365.0)
    fac = np.where(idx >= start_date, (1.0 + annual_rate) ** t, 1.0)
    return y * fac

# G) Event-Tage: Liste von Daten mit Multiplikator (±%)
def event_days(y: pd.Series, dates: list, mult=0.9):
    y = y.copy(); d = y.index.tz_convert("Europe/Berlin").date
    mask = pd.Index(d).isin({pd.to_datetime(x).date() for x in dates})
    y[mask] = y[mask] * float(mult)
    return y

# H) Netto-Last anwenden (PV/Wind abziehen)
def apply_net_load(y: pd.Series, pv_mw=0.0, wind_mw=0.0):
    pv = synthetic_pv(y.index, pv_mw) if pv_mw>0 else 0.0
    wd = synthetic_wind(y.index, wind_mw) if wind_mw>0 else 0.0
    return y - pv - wd
