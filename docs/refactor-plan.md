# Refactorplan Streamlit-App Stromlast-Forecast DE

Stand: 2026-04-24

## Zielbild

Die App soll von einem prototypischen Streamlit-Skript zu einer klar strukturierten Forecasting-Anwendung weiterentwickelt werden. Ziel ist eine fachlich belastbarere 24h-Stromlastprognose fuer Deutschland, eine besser wartbare Codebasis und ein reproduzierbarer Betrieb mit nachvollziehbaren Daten-, Modell- und Monitoring-Artefakten.

Der Refactor sollte nicht primaer kosmetisch sein. Der wichtigste Hebel ist die Trennung von UI, Datenzugriff, Feature Engineering, Forecasting, Validierung, Szenarien und Persistenz. Dadurch werden Modelllogik und Datenqualitaet testbar, waehrend Streamlit nur noch Orchestrierung und Darstellung uebernimmt.

## Aktueller Zustand

Die aktuelle App funktioniert als Demo und enthaelt bereits wichtige Bausteine:

- SMARD-Datenabruf fuer Netzlast Deutschland ueber `smard_data.py`.
- EDA mit Zeitraumwahl, Feiertagen, Vorjahresvergleich und Wochenprofilen in `app.py`.
- Baseline-Modelle, SARIMA-Forecast, Walk-Forward-Backtesting und Model Card in `forecast.py`.
- What-if-Szenarien in `scenarios.py`.
- Daily-Metrics-Job in `metrics_job.py` mit Artefaktpersistenz unter `artifacts/`.
- Docker-Setup und GitHub-Actions-Workflow fuer taegliche Metriken.

Gleichzeitig ist die Struktur stark gekoppelt:

- `app.py` enthaelt UI, Datenaufbereitung, Chart-Logik, Forecast-Orchestrierung und Szenario-Workflow in einer Datei.
- `forecast.py` mischt Modellfunktionen, Metriken, Streamlit-KPI-Komponenten, Artefaktpfade, JSON-Persistenz und Model-Card-Rendering.
- `smard_data.py` importiert UI-Helfer aus `forecast.py`; dadurch haengt Datenzugriff indirekt an Streamlit/UI.
- Viele Funktionen verwenden globale Pfade, implizite Zeitzonen und implizite Modellparameter.
- Fehler werden teilweise geschluckt, zum Beispiel im SARIMA-Backtesting.
- Es gibt keine automatisierten Tests fuer SMARD-Parsing, DST-Verhalten, Metriken, Backtesting oder Szenarien.

## Fachliche Refactor-Ziele

### 1. Datenmodell und Einheiten klaeren

Die App verwendet an mehreren Stellen `MW`, `MWh`, `Verbrauch MWh` und `total load (MWh)` uneinheitlich. SMARD-Netzlast ist eine Leistung in MW. Bei stuendlicher Aufloesung entspricht die Summe ueber Stunden einer Energie in MWh.

Massnahmen:

- Eine zentrale Definition fuer Zielvariable einfuehren: `load_mw`.
- Plot-Achsen fuer Zeitreihen als `Netzlast (MW)` beschriften.
- Energie-KPIs explizit als `MWh = Summe load_mw * 1h` berechnen.
- Metriken eindeutig benennen: `MAE_MW`, `sMAPE_pct`, `MASE_168h`.
- README, Model Card und UI-Texte auf diese Semantik angleichen.

### 2. Zeitzonen und DST robust machen

Deutschland hat DST-Wechsel. Die App konvertiert teilweise lokal, teilweise UTC-naiv. Das ist fuer SARIMA sinnvoll, muss aber als explizite Konvention dokumentiert und getestet werden.

Massnahmen:

- Eine zentrale Zeit-Konvention definieren:
  - Extern/UI: `Europe/Berlin`, tz-aware.
  - Modellintern: UTC-naiver, lueckenloser Stundenindex.
  - Persistenz: ISO-Zeitstempel mit Zeitzone oder UTC.
- Helper fuer Konvertierung einziehen, zum Beispiel `to_model_index()` und `to_display_index()`.
- Tests fuer Sommerzeit-Start und Sommerzeit-Ende ergaenzen.
- Keine direkte Nutzung von `asfreq("h")` ohne vorherige Konvention und Duplikatbehandlung.

### 3. Datenqualitaet als fachliche Vorbedingung behandeln

Der Forecast sollte nur dann als belastbar dargestellt werden, wenn die Eingangsdaten ausreichend frisch und vollstaendig sind.

Massnahmen:

- Datenqualitaets-Check aus der UI herausloesen und als Service implementieren.
- Quality-Objekt einfuehren mit `last_timestamp`, `freshness_lag_hours`, `missing_count`, `coverage_pct`, `duplicate_count`.
- Klare Statuslogik definieren:
  - `ok`: ausreichend frisch und vollstaendig.
  - `warning`: leichte Luecken oder SMARD-Lag.
  - `blocked`: Forecast nicht sinnvoll.
- Forecast-Buttons und Ops-Anzeige an diesen Status koppeln.

### 4. Backtesting fachlich vereinheitlichen

Aktuell existieren mehrere Backtesting-Pfade: Baselines, SARIMA rolling, aktuelles Modell und Daily-Metrics-Scoring. Diese sollten dieselbe Scoring-Logik nutzen.

Massnahmen:

- Einen zentralen Backtesting-Service schaffen.
- Einheitliche Fold-Definition dokumentieren:
  - Forecast-Horizont `H=24`.
  - Fold-Schritt `24h`.
  - Trainingsfenster konfigurierbar.
  - Mindestens 80 Prozent Coverage pro Fold.
- Alle Modelle gegen dieselbe Zielserie und dieselben Folds evaluieren.
- Baseline `seasonal_naive_168h` als primaere Referenz fuer Promotion-Gate festlegen.
- Fehlgeschlagene Folds protokollieren statt still zu ignorieren.

### 5. Modellstrategie erweitern

SARIMA ist als transparenter Start sinnvoll, aber fuer Lastprognosen sollten zusaetzliche robuste Baselines und Feature-basierte Modelle vorbereitet werden.

Massnahmen:

- Baselines als feste Mindestanforderung behalten:
  - letzte Stunde.
  - gleicher Vortag.
  - gleiche Vorwoche.
  - Drift nur optional, da fachlich fuer Last oft schwach.
- Kalenderfeatures zentralisieren:
  - Stunde, Wochentag, Wochenende, Feiertag, Monat, Brueckentag optional.
- Optionales naechstes Modell vorbereiten:
  - Gradient Boosting oder Random Forest mit Lag-Features.
  - Lag-Features: `t-24`, `t-48`, `t-168`, Rolling Means.
  - SARIMA als interpretierbarer Vergleich, nicht als einzige Modellklasse.
- Prediction Intervals getrennt bewerten:
  - Coverage.
  - mittlere Intervallbreite.

### 6. Szenarien klar von Prognosen trennen

Die Szenarien sind aktuell Post-Processing auf Historie und keine echte Prognose. Das ist fachlich legitim, muss aber im Code und UI klar bleiben.

Massnahmen:

- Szenario-Modul in `simulation` oder `what_if` umbenennen.
- Szenario-Ergebnisse als Simulation kennzeichnen.
- Validierungen fuer Parameterbereiche einfuehren.
- Lastverschiebung pro Tag energieerhaltend testen.
- Optional spaeter Szenarien auf Forecasts anwenden statt nur auf Historie.

## Architektur-Zielstruktur

Vorgeschlagene Paketstruktur:

```text
power_forecast_de/
  app/
    main.py
    pages/
      home.py
      eda.py
      forecast.py
      ops.py
      scenarios.py
    components/
      kpi.py
      charts.py
      layout.py
  src/
    power_forecast/
      config.py
      time.py
      data/
        smard_client.py
        quality.py
        schema.py
      features/
        calendar.py
        lagged.py
      models/
        baselines.py
        sarima.py
        registry.py
      evaluation/
        metrics.py
        backtesting.py
        gates.py
      scenarios/
        transforms.py
        kpis.py
      artifacts/
        store.py
        model_card.py
      ops/
        daily_metrics.py
  tests/
    test_smard_client.py
    test_time.py
    test_metrics.py
    test_backtesting.py
    test_scenarios.py
```

Falls das Repo bewusst klein bleiben soll, kann die Struktur kompakter starten:

```text
src/power_forecast/
  config.py
  smard.py
  quality.py
  features.py
  baselines.py
  sarima.py
  metrics.py
  backtesting.py
  scenarios.py
  artifacts.py
  ui_components.py
```

Wichtig ist nicht die Anzahl der Dateien, sondern die Richtung der Abhaengigkeiten:

```text
Streamlit UI -> Services -> Domain/Model/Data
```

Domain-, Modell- und Datenmodule duerfen kein Streamlit importieren.

## Konkrete Refactor-Schritte

### Phase 1: Entkopplung ohne Verhaltensaenderung

Ziel: Bestehende Funktionalitaet erhalten, aber technische Kopplung reduzieren.

- `forecast.py` in fachliche Module und UI-Komponenten aufteilen.
- KPI-HTML und CSS aus `forecast.py` und `smard_data.py` in ein UI-Komponentenmodul verschieben.
- `smard_data.py` von Streamlit und Forecast-UI entkoppeln.
- Wildcard-Imports in `app.py` entfernen.
- Globale Artefaktpfade in `config.py` zentralisieren.
- Logging statt stiller `except Exception: pass`-Bloecke einfuehren.
- `app.py` in Render-Funktionen je Tab zerlegen.

Akzeptanzkriterien:

- Die Streamlit-App startet weiterhin mit `streamlit run app.py`.
- Kein Nicht-UI-Modul importiert `streamlit`.
- Forecast, EDA, Ops und Szenarien zeigen dieselben Inhalte wie vorher.

### Phase 2: Tests und Datenkonvention

Ziel: Kritische fachliche Logik absichern.

- `pytest` als Dev-Abhaengigkeit ergaenzen.
- Tests fuer Metriken `smape`, `mase` und MAE-Scoring schreiben.
- Tests fuer Szenarien schreiben:
  - Feiertags-/Wochenendmultiplikator.
  - Energieerhaltung bei Load-Shift.
  - Event-Tage.
  - Effizienztrend.
- Tests fuer Zeitkonvertierung und DST-Faelle schreiben.
- SMARD-Client mit Mock-Responses testen.
- CI-Schritt fuer Tests ergaenzen.

Akzeptanzkriterien:

- `pytest` laeuft lokal ohne Netzwerkzugriff.
- SMARD-spezifische API-Tests nutzen Mocks oder Fixtures.
- DST-Sonderfaelle sind explizit abgedeckt.

### Phase 3: Einheitliches Forecasting und Backtesting

Ziel: Modellvergleich und Ops-Scoring auf dieselbe Grundlage stellen.

- Gemeinsame Fold-Generierung implementieren.
- Baselines und SARIMA ueber ein gemeinsames Interface laufen lassen.
- Daily-Metrics-Job auf denselben Evaluationsservice umstellen.
- Einheitliches Ergebnisformat definieren:
  - `model_name`
  - `forecast_issue_time`
  - `horizon`
  - `points_compared`
  - `MAE_MW`
  - `sMAPE_pct`
  - `MASE_168h`
  - `baseline_gain_pct`
- Promotion-Gate als konfigurierbare Regel implementieren.
- Model Card aus Artefakten generieren, nicht aus Streamlit-Session-State.

Akzeptanzkriterien:

- Backtesting in UI und Daily Job liefert vergleichbare Metriken.
- Fehlgeschlagene Folds sind im Ergebnis sichtbar.
- Baseline-Gain bezieht sich immer auf `seasonal_naive_168h`.

### Phase 4: Fachliche Modellverbesserung

Ziel: Prognoseguete verbessern und realistischere Unsicherheiten anzeigen.

- Lag-/Kalenderfeature-Pipeline implementieren.
- Feature-basiertes Modell als zusaetzlichen Kandidaten einfuehren.
- SARIMA-Parameter und Modellklasse versionieren.
- Rolling-Origin-Auswertung ueber mehrere Validierungsfenster ergaenzen.
- Intervalle evaluieren:
  - Coverage.
  - Intervallbreite.
  - Kalibrierung.
- Optional Wetterdaten als spaetere Erweiterung vorbereiten, aber nicht in diesen Refactor erzwingen.

Akzeptanzkriterien:

- Neues Modell muss in Backtesting gegen `seasonal_naive_168h` antreten.
- UI zeigt Modellvergleich, nicht nur Einzelergebnis.
- Model Card enthaelt Trainingsfenster, Featureliste, Datenstand und Validierung.

### Phase 5: UI/UX und Produktqualitaet

Ziel: Die App wirkt weniger wie ein Notebook und mehr wie ein operatives Tool.

- Sidebar nach Kontext strukturieren statt alle Controls global zu mischen.
- Forecast-Tab klar in drei Bereiche teilen:
  - Datenstatus.
  - Forecast-Ausgabe.
  - Validierung/Modellstatus.
- EDA-Charts in eigene Chart-Funktionen verschieben.
- Ops-Tab fuer historische Metriken aus `artifacts/metrics.csv` ausbauen.
- Szenarien mit Presets versehen, zum Beispiel `Feiertag`, `Demand Response`, `Effizienz`.
- Lange Modellberechnungen visuell sauber kapseln und Ergebnisse cachebar machen.

Akzeptanzkriterien:

- Jede UI-Seite hat eine klare Aufgabe.
- Teure Berechnungen laufen nur nach explizitem Klick.
- Nutzer sieht Datenstand, Modellstand und Validierungsstand ohne Modelldetails suchen zu muessen.

## Technische Schulden und konkrete Risiken

- `show_last_val()` nutzt `float(df.get("MAE", np.nan))`; bei DataFrame-Spalten ist das fragil und kann bei mehreren Zeilen fehlschlagen.
- `eval_sarimax_rolling90_fast()` gibt im Fehlerfall teilweise nur `out` zurueck, waehrend aufrufender Code zwei Werte erwartet.
- `requests.get()` im SMARD-Client nutzt nicht konsequent `raise_for_status()`.
- SMARD-API-Fehler, leere Serien und JSON-Formatabweichungen werden nicht sauber typisiert.
- Artefakt-Fallbacks zeigen auf Repo-Root-Dateien, waehrend vorhandene Artefakte unter `artifacts/` liegen.
- `metrics_job.py` erzeugt `base_loc=s_naive(s, ...)` mit lokalem Index-Verhalten, das gegen Forecast-Zeitstempel explizit getestet werden sollte.
- CSS fuer KPI-Cards ist dupliziert und wird mehrfach injiziert.
- `.pyc`, IDE-Dateien und erzeugte Artefakte sollten in `.gitignore`/Repo-Hygiene geprueft werden.

## Priorisierte Umsetzung

Empfohlene Reihenfolge:

1. UI-unabhaengige Kernmodule schaffen und Streamlit aus Daten-/Forecast-Modulen entfernen.
2. Zeit- und Einheitenkonvention festlegen, dokumentieren und testen.
3. Backtesting vereinheitlichen und Daily Job darauf umstellen.
4. Tests fuer Metriken, Szenarien, SMARD-Parsing und DST ergaenzen.
5. Feature-basiertes Modell als Kandidat hinzufuegen.
6. UI in Tabs/Komponenten aufraeumen und Ops-Ansicht ausbauen.

## Definition of Done fuer den Refactor

- Nicht-UI-Code ist ohne Streamlit importierbar und testbar.
- `pytest` deckt Kernlogik fuer Daten, Zeit, Metriken, Backtesting und Szenarien ab.
- Einheiten und Zeitzonen sind in Code, UI und README konsistent.
- Forecasting, Backtesting und Daily-Metrics-Job nutzen gemeinsame Services.
- Artefakte sind versioniert, reproduzierbar und mit Model Card nachvollziehbar.
- Die App zeigt dem Nutzer transparent: Datenstand, Modellstand, Forecast, Unsicherheit und historische Modellguete.

