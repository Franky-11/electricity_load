import pandas as pd, requests
import streamlit as st

from forecast import kpi_card_2

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


    # 2) Alle Blöcke holen & zusammenfügen
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




def show_data_quality(s, tz="Europe/Berlin",last=90):
    # 1) Index lokal & stündlich harmonisieren
    idx = s.index
    if idx.tz is not None:
        idx_local = idx.tz_convert(tz)
        y = s.tz_convert("UTC").tz_localize(None).asfreq("h")  # intern UTC-naiv
    else:
        idx_local = idx.tz_localize(tz)
        y = s.asfreq("h")
    last_local = idx_local.max()
    now_local = pd.Timestamp.now(tz)

    # 2) Kennzahlen
    lag_h = (now_local - last_local).total_seconds() / 3600
    w_start = last_local - pd.Timedelta(days=last)
    y_last = y.loc[w_start.tz_convert("UTC").tz_localize(None): last_local.tz_convert("UTC").tz_localize(None)]
    expected = last * 24
    missing_last = int(y_last.isna().sum())
    cover_last = 100 * (1 - missing_last / max(expected, 1))
    dups_total = int(s.index.duplicated(keep="first").sum())
    tz_name = str(idx_local.tz)

    # 3) UI
    st.markdown("""
        <style>
        :root{
          --bg:#0b132b; --card:#0f1c2e; --border:#133046;
          --text:#d8f3ff; --muted:#7dd3fc; --foot:#9cc3d5;
          --pos:#22c55e; --neg:#ef4444;
        }
        .kpi{
          background: linear-gradient(180deg, var(--card), var(--bg));
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 16px 18px;
          box-shadow: 0 10px 30px rgba(0,0,0,.25);
        }
        .kpi-head{
          font-size: 14px; color: var(--muted);
          display:flex; align-items:center; gap:8px; letter-spacing:.3px;
        }
        .kpi-ic{ font-size:16px; }
        .kpi-val{
          margin-top:6px; line-height:1.1;
          font-size: 32px; font-weight: 700; color: var(--text);
        }
        .kpi-unit{ font-size:16px; color: var(--muted); margin-left:6px; }
        .kpi-foot{ font-size:12px; color: var(--foot); margin-top:6px; }
        .kpi.pos .kpi-val{ text-shadow:0 0 12px rgba(34,197,94,.35); }
        .kpi.neg .kpi-val{ text-shadow:0 0 12px rgba(239,68,68,.35); }
        .kpi.neu .kpi-val{ text-shadow:0 0 10px rgba(125,211,252,.25); }
        </style>
        """, unsafe_allow_html=True)

    st.subheader("🧪 Data Quality & Freshness")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card_2("Letzter Datenpunkt", last_local.strftime("%Y-%m-%d %H:%M"),
                 unit="", icon="⚡", footnote="")

    with c2:
        kpi_card_2("Frische-Lag", f"{lag_h:.1f} h",
                   unit="", icon="🕑", footnote="")
    with c3:
        kpi_card_2(f"Missing ({last}T)", f"{missing_last} / {expected} ({cover_last:.1f}%)",
                   unit="", icon="🔎", footnote="")
    with c4:
        kpi_card_2("Duplikate (ges.)", dups_total,
                   unit="", icon="‼️", footnote="")

    #c2.metric("Frische-Lag", f"{lag_h:.1f} h")
    #c3.metric(f"Missing ({last}T)", f"{missing_last} / {expected} ({cover_last:.1f}%)")
   # c4.metric("Duplikate (ges.)", dups_total)

    #if lag_h > 2 or missing_last > 0 or dups_total > 0:
    #    st.warning("Auffälligkeiten erkannt. Hinweis: SMARD kann verzögert liefern. Quelle evtl. verzögert.")
      #  if st.button("🔄 Re-Load Daten"):
         #   st.experimental_rerun()

    st.caption(f"TZ: {tz_name} | Frequenz: stündlich | DST wird automatisch berücksichtigt")
