# Stromlast‑Forecast DE
> Interaktive Web‑App für **EDA**, **Baselines/Backtesting**, optional **SARIMA(+exog)** und **What‑if‑Szenarien** auf **stündlichen** Stromlastdaten (TZ: *Europe/Berlin*).

---

## 🔗 Live‑App

Füge hier deinen Link ein (z. B. Streamlit Community Cloud):

**App:** https\://\<DEIN‑APP‑LINK>

---

## ✨ Features

- **EDA**: Zeitraumwahl, Feiertags‑Overlay, optional Vorjahresvergleich; Linien‑ & Wochenmusterplots.
- **Baselines & Backtesting**: Naive / Seasonal‑Naive (m=24/168) / Drift; Walk‑Forward mit fixem Horizont.
- **Metriken**: **MAE**, **sMAPE**, **MASE (168h)**.
- **SARIMA (+exogene Kalender‑Features)**: optional, inkl. (Beispiel‑)PI‑Bänder via CSV.
- **What‑if Szenarien** (Post‑Processing, extrem schnell):
  - Feiertag/Weekend‑Multiplikator,
  - Peak→Off‑Peak‑Verschiebung (Demand Response),
  - Temperatur‑Sensitivity (linear),
  - Effizienz‑/Spar‑Trend (ab Datum),
  - Event‑Tage (±%),
  - Netto‑Last via **synthetische PV/Wind**.
- **KPI‑Cards**: Δ Peak (MW), Δ Energie (MWh) für Szenarien.

---

## 🗂️ Projektstruktur

```
app.py         # UI: Sidebar (EDA, Backtesting), Forecast‑Tab, Szenario‑Panel, KPIs
forecast.py    # Daten‑Loader, Baselines, Walk‑Forward, Metriken, (optionales) SARIMA
scenarios.py   # What‑if‑Helperfunktionen (Holiday/Weekend, Shift, Temp, Effizienz, Event, PV/Wind)
requirements.txt
```
---

## 🧠 Modelle & Metriken

- **Baselines:**
  - `naive`  → - `naive` → $\hat{y}_{t+h} = y_t$
  - `seasonal_naive(m)`  → $\hat{y}_{t+h} = y_{t+h-m}$
  - `drift`  → trend per letzter Änderung
- **SARIMA** (optional): kleiner, stabiler Suchraum mit saisonaler Woche (`m=168`); exog: Wochenende/Feiertag.
- **Metriken:**
  - **MAE** (robust),
  - **sMAPE** (prozentual; Achtung bei Werten \~0),
  - **MASE‑168h** (vergleichbar über Skalen).



---

## 🧪 What‑if‑Szenarien (Post‑Processing)

Szenarien ändern **Forecasts oder Historie** nachträglich (keine Modell‑Refits):

- **Holiday/Weekend‑Multiplikator**: skaliert nur diese Tage.
- **Load‑Shift**: verschiebt x% von Peak‑ in Off‑Peak‑Stunden (Energieerhaltung pro Tag).
- **Temperatur‑Sensitivity**: ±% je °C Abweichung (linear, synthetisch).
- **Effizienz‑Trend**: ab Datum pro Jahr ±r% (multiplikativ).
- **Event‑Tage**: Liste von Datumswerten ±%.
- **Netto‑Last (PV/Wind)**: synthetische Profile werden abgezogen.

> **Interpretation:** Δ‑KPIs zeigen **Peak‑Reduktion** (MW) & **Energie‑Δ** (MWh). Ergebnisse sind **Simulationen**, keine Prognosen.

---