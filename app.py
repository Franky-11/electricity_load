import streamlit as st

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from holidays import Germany

from forecast import*
from scenarios import*
from smard_data import load_smard_api

pio.templates.default = "plotly_dark"


st.set_page_config(layout="wide")

st.sidebar.header("EDA")



show_holidays = st.sidebar.checkbox("Feiertage markieren", value=True)

# ---------- Vorjahres-Vergleich (Checkbox) ----------
compare_prev = st.sidebar.checkbox("Mit Vorjahr vergleichen (gleicher Zeitraum)", value=False)  # ← NEU

# Auswahl anwenden (datumsgenau inkl. Endtag)

# EDA für stündliche Serie s (tz=Europe/Berlin, freq="h"
s=load_smard_api(years=2)
df = s.to_frame("y")
idx=df.index

df["dow"]=idx.dayofweek
df["hour"]=idx.hour
de_hol= Germany(years=range(idx.year.min(), idx.year.max() + 1))
df["is_hol"]=pd.Series(idx.date,index=idx).map(lambda d: d in de_hol)
df["h_w"] = df["dow"]*24 + df["hour"]
df["month"]=idx.month
# ---------- Sidebar: Datumsbereich ----------
min_date = df.index.date.min()
max_date = df.index.date.max()

st.sidebar.markdown("**Zeitraum**")
date_range = st.sidebar.date_input(
    "Von ... bis ... (inklusive)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date)

# Normalisieren & validieren
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    # Falls Nutzer nur ein Datum wählt, nimm das als beides
    start_date = end_date = date_range

if start_date > end_date:
    st.sidebar.warning("Startdatum > Enddatum – ich habe es getauscht.")
    start_date, end_date = end_date, start_date


mask = (df.index.date >= start_date) & (df.index.date <= end_date)
df_view = df.loc[mask].copy()

# Vorjahresfenster berechnen  #
prev_info = ""
df_prev = pd.DataFrame(columns=df.columns)
if compare_prev:

    start_ts = pd.Timestamp(start_date )
    end_ts = pd.Timestamp(end_date)
    prev_start = (start_ts - pd.DateOffset(years=1)).date()
    prev_end = (end_ts - pd.DateOffset(years=1)).date()
    mask_prev = (df.index.date >= prev_start) & (df.index.date <= prev_end)
    df_prev = df.loc[mask_prev].copy()
    if df_prev.empty:
        st.sidebar.info("Kein passender Vorjahreszeitraum in den Daten.")
    else:
        prev_info = f" · Vorjahr: {prev_start} bis {prev_end}"


#===========================================================================#

st.title("⚡ Stromverbrauch Deutschland")
welcome,eda,forecast=st.tabs(["Home","EDA","Forecast"])

with welcome:
    left,space, right = st.columns([1.25,0.2, 1])
    st.write("")
    with left:
        abruf=pd.Timestamp.now().strftime("%Y-%m-%d")
        st.subheader("Willkommen! 👋")
        st.markdown(
            f"""
            **Diese App** zeigt Trend, Wochenmuster und saisonale Mittelwerte für den deutschen Stromverbrauch

            **Features**
            - Zeitraum in der Sidebar wählen (inkl. Vorjahresvergleich).
            - Trend mit **Rolling 24h/7d** und **Feiertagsmarkierung** ansehen.
            - **Wochenprofil** und **Wochentage/Monate** vergleichen.
            - **Backtesting & Forecasting** 24h Vorhersage der Lastdaten & Vergleich verschiedener Modelle

            **Datensatz:**  
            Daten: Bundesnetzagentur | SMARD.de – Bereich „Stromverbrauch: Gesamt (Netzlast)“, Region DE, Abruf: {abruf}, Lizenz: CC BY 4.0.
            
            """
        )

    with right:
        st.image("app_picture.png")

with eda:
    # 1) Trendblick

    st.subheader("Trend & Wochenmuster (EDA)")
    st.caption(f"TZ: Europe/Berlin · freq=H · Auswahl: {start_date} bis {end_date}{prev_info}")

    # ---------- Chart 1: Trend + Rolling Means + Feiertage ----------
    rm24 = df_view["y"].rolling(24, min_periods=1).mean()
    rm168 = df_view["y"].rolling(168, min_periods=1).mean()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_view.index, y=df_view["y"], mode="lines",
                          name="y", line=dict(width=1, color="grey")))
    fig1.add_trace(go.Scatter(x=df_view.index, y=rm24, mode="lines",
                          name="RM 24h", line=dict(width=2,color="blue")))
    fig1.add_trace(go.Scatter(x=df_view.index, y=rm168, mode="lines",
                          name="RM 7d", line=dict(width=2, dash="dot")))

    if show_holidays:
        hol_days = sorted(set(df_view.index.date[df_view["is_hol"]]))
    for d in hol_days:
        x0 = pd.Timestamp(d, tz="Europe/Berlin")
        x1 = x0 + pd.Timedelta(days=1)
        fig1.add_vrect(x0=x0, x1=x1, fillcolor="green", opacity=0.2, line_width=0, layer="below")

    fig1.update_layout(
    title="Serie + Rolling Means + Feiertage",
    height=380, hovermode="x unified", margin=dict(l=40, r=20, t=60, b=30),
    )
    fig1.update_yaxes(title_text="Verbrauch MWh")

    st.plotly_chart(fig1, use_container_width=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        # ---------- Chart 2: Wochenmuster (Hour-of-Week 0..167) ----------
        prof = (df_view.dropna(subset=["y"])
                .groupby("h_w")["y"].mean()
                .reindex(range(168)))  # 0..167

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=prof.index, y=prof.values, mode="lines", name="Aktuell"))

        if compare_prev and not df_prev.empty:  # ← NEU
            prof_prev = (df_prev.dropna(subset=["y"])
                         .groupby("h_w")["y"].mean()
                         .reindex(range(168)))
            fig2.add_trace(go.Scatter(x=prof_prev.index, y=prof_prev.values, mode="lines",
                                      name="Vorjahr (selber Zeitraum)", line=dict(dash="dash")))

        fig2.update_layout(title="Mittelwert je Stunde der Woche (0..167)",
                           height=320, margin=dict(l=40, r=20, t=60, b=30), hovermode="x unified",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        fig2.update_yaxes(title_text="Verbrauch MWh je Stunde")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # ---------- Chart 3: Mittelwert je Wochentag ----------
        dow_mean = (df_view.dropna(subset=["y"])
                    .groupby("dow")["y"].mean()
                    .reindex(range(7)))
        dow_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=dow_names, y=dow_mean.values, name="Aktuell", offsetgroup=0))

        if compare_prev and not df_prev.empty:  # ← NEU
            dow_prev = (df_prev.dropna(subset=["y"])
                        .groupby("dow")["y"].mean()
                        .reindex(range(7)))
            fig3.add_trace(go.Bar(x=dow_names, y=dow_prev.values, name="Vorjahr", offsetgroup=1))

        fig3.update_layout(title="Mittelwert je Wochentag", barmode="group",
                           height=320, margin=dict(l=40, r=20, t=60, b=30))
        fig3.update_yaxes(title_text="Verbrauch MWh je Stunde")
        st.plotly_chart(fig3, use_container_width=True)

    with col3:
        # ---------- Chart 4: Mittelwert je Monat ----------
        month_mean = (df_view.groupby("month")["y"].mean()
                      .reindex(range(1, 13)))
        month_names = ["Jan","Feb","Mrz","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]
        xmonths = list(range(1, 13))

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=xmonths, y=month_mean.values, fill="tozeroy",
                                  name="Aktuell", mode="lines"))

        if compare_prev and not df_prev.empty:  # ← NEU
            month_prev = (df_prev.groupby("month")["y"].mean()
                          .reindex(range(1, 13)))
            fig4.add_trace(go.Scatter(x=xmonths, y=month_prev.values,
                                      name="Vorjahr", mode="lines", line=dict(dash="dash")))

        fig4.update_xaxes(tickmode="array", tickvals=xmonths, ticktext=month_names)
        fig4.update_layout(title="Mittelwert je Monat", height=320, margin=dict(l=40, r=20, t=60, b=30))
        fig4.update_yaxes(title_text="Verbrauch MWh je Stunde")
        st.plotly_chart(fig4, use_container_width=True)


with forecast:
    st.subheader("Walk-Forward Backtesting")
    ph = st.empty()
    ph.info("Starte Validierung in der Sidebar")

    st.sidebar.divider()
    st.sidebar.header("Walk-Forward Backtesting")

    with st.sidebar.form("Forecast"):
        #st.markdown("Walk-Forward Backtesting")
        win_days=st.slider("Trainingsfenster (Rolling Window)",30,90,30,5)
        eval_days=st.number_input("Anzahl Validierungstage",10,30,30,5)
        with_sarimax=st.checkbox("SARIMA+exog.Features testen?",value=False)
        submit=st.form_submit_button("Validierung starten")

    if submit:
        st.session_state["backtesting"] = True

    if st.session_state.get("backtesting", False):

        ph.write("")
        with st.spinner("Walk-Forward Backtesting Baslines..."):
            df_base = eval_baselines(s, H=24, m=24, win_days=win_days, eval_days=eval_days)
            st.markdown(f"**Baselines** | {win_days} Tage Trainingsfenster, {eval_days} Tage Validierung")
            st.dataframe(df_base)
        if with_sarimax:
            with st.spinner("Walk-Forward Backtesting SARIMA..."):
                df_sarimax = eval_sarimax_rolling90_fast(s, H=24, window_days=win_days, days_to_eval=eval_days,
                                                         step_hours=24)
                st.markdown(f"**SARIMA + exog. Features** | {win_days}Tage Trainingsfenster, {eval_days} Tage Validierung")
                st.caption("order (1,0,0) x seasonal_order (0,1,0,168)")
                st.dataframe(df_sarimax)
        else:
           st.markdown("Letzte Validierung **SARIMA + exog. Features**")
           df_sarimax=pd.DataFrame([[645,1.5,0.29,95.2]],columns=["MAE","sMAPE (%)","MASE_168","coverage PI (%)"])
           st.caption("order (1,0,0) x seasonal_order (0,1,0,168) ")
           st.caption("Validierung am 22.09.2025 | 90 Tage Trainingsfenster, 30 Tage Validierung")
           st.dataframe(df_sarimax,hide_index=True)

        last_week_fc = s_naive(s, h=24, m=168)
        last_24h_fc = s_naive(s, h=24, m=24)
        #sarima_pred=load_sarima_pred()

        # Nur die letzten 3 Tage der historischen Daten für den Plot nehmen
        hist = df.iloc[-3 * 24:]

        # Plot erstellen
        fig5 = px.area(hist, x=hist.index, y="y", title="Forecast 24h vs. Historie")

        # Traces für die Forecasts hinzufügen
        fig5.add_trace(go.Scatter(x=last_24h_fc.index, y=last_24h_fc.values, name="naive_24h", mode="lines"))
        fig5.add_trace(go.Scatter(x=last_week_fc.index, y=last_week_fc.values, name="naive_168h", mode="lines"))

        # Sarima Forecast aus csv
       # fig5.add_trace(go.Scatter(x=sarima_pred.index, y=sarima_pred["yhat"], name="SARIMA", mode="lines",line=dict(color="white", width=2, dash="solid")))
        # 1) Prediction-Band
        #fig5.add_trace(go.Scatter(
          #  x=sarima_pred.index, y=sarima_pred["lo"],
         #   mode='lines',
         #   line=dict(width=0),  # keine sichtbare Linie unten
         #   showlegend=False,
         #   hoverinfo='skip',
         #   name='PI lower'
      #  ))
       # fig5.add_trace(go.Scatter(
        #    x=sarima_pred.index, y=sarima_pred["hi"],
        #    mode='lines',
        #    line=dict(width=0),
        #    fill='tonexty',  # füllt bis zum vorherigen Trace (y_lower)
        #    fillcolor='rgba(211, 211, 211, 0.5)',  # halbtransparent
        #    name='Prediction interval SARIMA'
        #))


        forecast_start_time = last_24h_fc.index[0]

        # vertikale Linie als "shape" hinzufügen
        fig5.add_shape(
            type="line",
            x0=forecast_start_time, y0=0,
            x1=forecast_start_time, y1=1,
            yref="paper",  # y-Koordinaten beziehen sich auf die Höhe des Plots (0=unten, 1=oben)
            line=dict(color="Red", width=2, dash="dash")
        )

        fig5.add_annotation(
            x=forecast_start_time,
            y=1.05,  # Leicht über dem oberen Rand des Plots
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            xanchor="left"
        )

        # Layout und Labels verbessern
        fig5.data[0].name = "Historie"  # Label für die Area-Trace in der Legende
        fig5.update_xaxes(title_text="")
        fig5.update_yaxes(title_text="total load (MW)")
        fig5.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )

        st.plotly_chart(fig5, use_container_width=True)

    #======================================================================================================#

    #Szenarien
    st.divider()
    st.subheader("🔬 Szenario-Simulation (What-if)")
    rng = st.date_input("Historienfenster", [start_date, end_date])
    if isinstance(rng, tuple) and len(rng) == 2:
        start_date, end_date = rng
    else:
        start_date = end_date = rng
    s_loc = s.tz_convert("Europe/Berlin")
    y_base = s_loc[(s_loc.index.date >= start_date) & (s_loc.index.date <= end_date)].asfreq("h")
    y_base.name = "History"

    # 2) Szenario-Panel

    st.markdown("### 🎛️ Szenarien")
    col1, col2 = st.columns(2)

    with col1:
        use_hw = st.checkbox("Feiertag/Wochenende skalieren")
        use_shift = st.checkbox("Peak → Off-Peak verschieben")
        use_temp = st.checkbox("Temperatur-Sensitivität")
        use_eff = st.checkbox("Effizienz-Trend (ab Datum)")
    with col2:
        use_event = st.checkbox("Event-Tag(e) skalieren")
        use_pv = st.checkbox("PV berücksichtigen (Netto-Last)")
        use_wind = st.checkbox("Wind berücksichtigen (Netto-Last)")

    y_scn = y_base.copy()

    # Parameter + Anwendung
    if use_hw:
        hol_mult = st.slider("Feiertags-Multiplikator", 0.6, 1.2, 0.9, 0.01)
        we_mult = st.slider("Wochenend-Multiplikator", 0.6, 1.2, 0.95, 0.01)

        years = range(int(y_scn.index.year.min()), int(y_scn.index.year.max()) + 1)
        holi = set(Germany(years=years).keys())
        #Auf den y_scn-Zeitraum filtern
        y_scn_dates = set(y_scn.index.date)  # Menge von datetime.date im Szenario
        holidays_in_range = holi  & y_scn_dates
        y_scn = mult_holiday_weekend(y_scn, holidays_in_range, hol_mult, we_mult)

    if use_shift:
        frac = st.slider("Verschiebe-Anteil", 0.0, 0.5, 0.1, 0.01)
        src = st.multiselect("Peak-Stunden (Quelle)", list(range(24)), default=[18, 19, 20, 21])
        dst = st.multiselect("Off-Peak-Stunden (Ziel)", list(range(24)), default=[2, 3, 4, 5])
        y_scn = shift_load(y_scn, frac, src, dst)

    if use_temp:
        delta = st.slider("Δ Temperatur vs. Referenz (°C)", -10, 10, 0)
        kpc = st.slider("Sensitivität (%/°C)", 0.0, 5.0, 1.5, 0.1) / 100.0
        y_scn = temp_adjust(y_scn, delta, kpc, mode="multiplicative")

    if use_eff:
        start = st.date_input("Effizienz ab", value=pd.to_datetime(y_scn.index.min()).date())
        rate = st.slider("Jährliche Änderung (%)", -10.0, 10.0, -1.0, 0.1) / 100.0
        y_scn = efficiency_trend(y_scn, start, rate)

    if use_event:
        dates = st.text_input("Event-Tage (YYYY-MM-DD, komma-getrennt)", "")
        mult = st.slider("Event-Multiplikator", 0.5, 1.5, 0.9, 0.01)
        dlist = [d.strip() for d in dates.split(",") if d.strip()]
        if dlist: y_scn = event_days(y_scn, dlist, mult)

    pv_mw = wind_mw = 0.0
    if use_pv:   pv_mw = st.number_input("PV-Kapazität (MW)", 0.0, 100000.0, 2000.0, 100.0)
    if use_wind: wind_mw = st.number_input("Wind-Kapazität (MW)", 0.0, 100000.0, 3000.0, 100.0)
    if use_pv or use_wind:
        y_scn = apply_net_load(y_scn, pv_mw, wind_mw)

    # 3) KPIs
    # === 1) Styles + Helper ===
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



    # === 2) Zwei KPI-Cards nebeneinander ===
    c1, c2 = st.columns(2)

    with c1:
        kpi_card("Δ Peak (vs. Base)", float(y_scn.max() - y_base.max()),
                 unit="MW", icon="⚡", footnote="Negativ = Peak-Reduktion")

    with c2:
        kpi_card("Δ Energie (Szenario − Base)", float((y_scn - y_base).sum()),
                 unit="MWh", icon="📉", footnote="1h-Auflösung → Summe=MWh")


    # 4) Plot
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=y_base.index, y=y_base, mode="lines", name=y_base.name))
    fig6.add_trace(go.Scatter(x=y_scn.index, y=y_scn, mode="lines", name="Scenario"))
    if use_hw:
        for d in holidays_in_range:
            x0 = pd.Timestamp(d, tz="Europe/Berlin")
            x1 = x0 + pd.Timedelta(days=1)
            fig6.add_vrect(x0=x0, x1=x1, fillcolor="green", opacity=0.2, line_width=0, layer="below")


    fig6.update_layout(margin=dict(l=40, r=20, t=30, b=20), xaxis_title="", yaxis_title="MW",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig6, use_container_width=True)
