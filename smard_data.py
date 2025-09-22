import pandas as pd, requests
import streamlit as st

@st.cache_data(show_spinner=True)
def load_smard_api(
    start=None, end=None, years=1,
    filter_id="410",   # 410 = Stromverbrauch: Gesamt (Netzlast)
    region="DE", res="hour",
    tz="Europe/Berlin"
):
    """Lädt Netzlast (MW) von SMARD per API.
    Zeitraum: [start, end] oder (wenn None) 'letzte <years> Jahre bis jetzt'.
    Gibt eine stündliche, tz-aware Serie (Europe/Berlin) zurück.
    """
    base = "https://www.smard.de/app/chart_data"
    now = pd.Timestamp.now()
    if end is None:   end = now
    if start is None: start = end - pd.DateOffset(years=years)
    start = pd.Timestamp(start,tz=tz); end = pd.Timestamp(end,tz=tz)

    # 1) Index mit verfügbaren Blöcken holen
    idx_url = f"{base}/{filter_id}/{region}/index_{res}.json"

    idx = requests.get(idx_url, timeout=30)
    #idx.raise_for_status()
    stamps = sorted(idx.json()["timestamps"])
    if not stamps:
        raise ValueError("SMARD: keine Zeitblöcke gefunden.")

    # --- 2) Nur relevante Blöcke auswählen (Überdeckung des Zielintervalls) ---
    stamp_ts = pd.to_datetime(stamps, unit="ms", utc=True)
    # Block-Ende = nächster Block-Start, letzter Block endet 'in der Zukunft'
    block_starts = list(stamp_ts)
    block_ends = list(stamp_ts[1:]) + [pd.Timestamp.utcnow()+ pd.Timedelta(days=7)]

    wanted = [int(s) for s, b0, b1 in zip(stamps, block_starts, block_ends)
              if (b1 >= start) and (b0 <= end)]

    if not wanted:
        raise ValueError("SMARD: kein Block überschneidet den gewünschten Zeitraum.")


    # 2) Alle Blöcke holen & zusammenfügen (ein Jahr ≈ überschaubare Anzahl)
    ser = []
    for ts in wanted:
        url = f"https://www.smard.de/app/chart_data/{filter_id}/{region}/{filter_id}_{region}_{res}_{ts}.json"
        data = requests.get(url, timeout=30).json()
        pairs = data.get("series", [])
        if not pairs:
            continue
        s = pd.Series(
            {pd.to_datetime(t, unit="ms", utc=True): v for t, v in pairs},
            dtype="float64"
        )
        ser.append(s)

    if not ser:
        raise ValueError("SMARD: keine Daten empfangen.")


    s = pd.concat(ser).sort_index()
    s = s[~s.index.duplicated(keep="last")]          # Duplikate zwischen Blöcken

    if pd.infer_freq(s.index) not in ("H", "h"):
        s = s.resample("H").mean()  # 15-min → Stunde: MW mitteln
    else:
        s = s.asfreq("h")  # sorgt für lückenloses Stundenraster

    # Trailing NaNs (unveröffentlichte letzte Stunde(n)) robust trimmen
    lv = s.last_valid_index()
    if lv is not None and lv < s.index[-1]:
        s = s.loc[:lv]

    # Zeitraumfilter in UTC anwenden
    start_utc = start.tz_convert("UTC")
    end_utc = end.tz_convert("UTC")
    s = s[(s.index >= start_utc) & (s.index <= end_utc)]

    # Zurück nach Europe/Berlin für die App
    s = s.tz_convert(tz).rename("Netzlast_MW")

    return s


