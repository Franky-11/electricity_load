import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from holidays import Germany

from power_forecast.charts import eda_month_chart, eda_trend_chart, eda_week_profile_chart, eda_weekday_chart
from power_forecast.config import METRICS_CSV, TIMEZONE
from power_forecast.forecast import (
    backtest_current_model,
    eval_baselines,
    eval_feature_model,
    eval_sarimax_rolling90_fast,
    forecast_from_params,
    load_validation_json,
    model_card_markdown,
    model_card_meta,
    refit_predict_pi,
    refit_predict_feature_model,
    resolve_params_spec,
    s_naive,
    save_validation_json,
    to_local,
)
from power_forecast.scenarios import apply_net_load, efficiency_trend, event_days, mult_holiday_weekend, shift_load, temp_adjust
from power_forecast.smard_data import load_smard_api
from power_forecast.ui_components import kpi_card, kpi_card_value, show_data_quality, show_last_val

pio.templates.default = "plotly_dark"


@st.cache_data(show_spinner=True)
def load_app_data(years: int = 2) -> pd.Series:
    return load_smard_api(years=years)


def prepare_eda_frame(s: pd.Series) -> pd.DataFrame:
    df = s.to_frame("y")
    idx = df.index
    df["dow"] = idx.dayofweek
    df["hour"] = idx.hour
    de_hol = Germany(years=range(idx.year.min(), idx.year.max() + 1))
    df["is_hol"] = pd.Series(idx.date, index=idx).map(lambda d: d in de_hol)
    df["h_w"] = df["dow"] * 24 + df["hour"]
    df["month"] = idx.month
    return df


def read_sidebar_filters(df: pd.DataFrame) -> tuple[bool, bool, pd.Timestamp.date, pd.Timestamp.date, pd.DataFrame, pd.DataFrame, str]:
    st.sidebar.header("EDA")
    show_holidays = st.sidebar.checkbox("Feiertage markieren", value=True)
    compare_prev = st.sidebar.checkbox("Mit Vorjahr vergleichen (gleicher Zeitraum)", value=False)

    min_date = df.index.date.min()
    max_date = df.index.date.max()
    st.sidebar.markdown("**Zeitraum**")
    date_range = st.sidebar.date_input(
        "Von ... bis ... (inklusive)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    if start_date > end_date:
        st.sidebar.warning("Startdatum > Enddatum – ich habe es getauscht.")
        start_date, end_date = end_date, start_date

    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    df_view = df.loc[mask].copy()

    prev_info = ""
    df_prev = pd.DataFrame(columns=df.columns)
    if compare_prev:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        prev_start = (start_ts - pd.DateOffset(years=1)).date()
        prev_end = (end_ts - pd.DateOffset(years=1)).date()
        mask_prev = (df.index.date >= prev_start) & (df.index.date <= prev_end)
        df_prev = df.loc[mask_prev].copy()
        if df_prev.empty:
            st.sidebar.info("Kein passender Vorjahreszeitraum in den Daten.")
        else:
            prev_info = f" · Vorjahr: {prev_start} bis {prev_end}"

    return show_holidays, compare_prev, start_date, end_date, df_view, df_prev, prev_info


def render_home() -> None:
    left, _, right = st.columns([1.25, 0.2, 1])
    st.write("")
    with left:
        abruf = pd.Timestamp.now().strftime("%Y-%m-%d")
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
        st.image("images/app_picture.png")


def render_eda(
    df_view: pd.DataFrame,
    df_prev: pd.DataFrame,
    start_date,
    end_date,
    prev_info: str,
    show_holidays: bool,
    compare_prev: bool,
) -> None:
    st.subheader("Trend & Wochenmuster (EDA)")
    st.caption(f"TZ: {TIMEZONE} · freq=H · Auswahl: {start_date} bis {end_date}{prev_info}")

    st.plotly_chart(eda_trend_chart(df_view, show_holidays=show_holidays), width="stretch")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(eda_week_profile_chart(df_view, df_prev, compare_prev), width="stretch")

    with col2:
        st.plotly_chart(eda_weekday_chart(df_view, df_prev, compare_prev), width="stretch")

    with col3:
        st.plotly_chart(eda_month_chart(df_view, df_prev, compare_prev), width="stretch")


def render_forecast(s: pd.Series, df: pd.DataFrame) -> None:
    st.sidebar.header("Forecast")
    refit_days = int(st.sidebar.segmented_control("Trainingsfenster", [30, 60, 90], default=60, key="forecast_refit_days"))

    st.sidebar.divider()
    st.sidebar.header("Validierung")
    with st.sidebar.form("Forecast"):
        win_days = st.slider("Trainingsfenster Backtest", 30, 90, 30, 5)
        eval_days = st.number_input("Validierungstage", 1, 60, 30, 1)
        with_feature_model = st.checkbox("Feature-Modell testen", value=True)
        with_sarimax = st.checkbox("SARIMA testen", value=False)
        submit = st.form_submit_button("Validierung starten")

    st.sidebar.divider()
    st.sidebar.header("Modellstatus")
    q_eval_days = int(st.sidebar.segmented_control("Qualitätsfenster", [1, 2, 3], default=1, key="qcheck_days"))

    st.subheader("Datenstatus")
    show_data_quality(s)

    if submit:
        st.session_state["backtesting"] = True
        for key in [
            "df_base",
            "feature_val_df",
            "feature_val_gain",
            "feature_val_details",
            "last_val_df",
            "last_val_gain",
            "last_val_meta",
        ]:
            st.session_state.pop(key, None)
        with st.spinner("Walk-Forward Backtesting Baselines..."):
            st.session_state["df_base"] = eval_baselines(s, H=24, m=24, win_days=win_days, eval_days=eval_days)

        if with_feature_model:
            with st.spinner("Walk-Forward Backtesting Feature-Modell..."):
                df_feature, feature_gain, feature_details = eval_feature_model(
                    s,
                    H=24,
                    win_days=win_days,
                    eval_days=eval_days,
                    step_hours=24,
                )
                st.session_state["feature_val_df"] = df_feature
                st.session_state["feature_val_gain"] = feature_gain
                st.session_state["feature_val_details"] = feature_details

        if with_sarimax:
            with st.spinner("Walk-Forward Backtesting SARIMA..."):
                df_sarimax, gain = eval_sarimax_rolling90_fast(s, H=24, window_days=win_days, days_to_eval=eval_days, step_hours=24)
                meta = {
                    "validated_at": pd.Timestamp.now(tz=TIMEZONE).strftime("%Y-%m-%d %H:%M"),
                    "order": "(1,0,0)",
                    "seasonal_order": f"(0,1,0,{168})",
                    "train_window_days": int(win_days),
                    "eval_days": int(eval_days),
                    "H": 24,
                }
                try:
                    save_validation_json(df_sarimax, gain, meta)
                except Exception as e:
                    st.warning(f"Speicherung fehlgeschlagen: {e}")
                st.session_state["last_val_df"] = df_sarimax
                st.session_state["last_val_gain"] = gain
                st.session_state["last_val_meta"] = meta

    st.divider()
    st.subheader("Forecast-Ausgabe")
    last_week_fc = s_naive(s, h=24, m=168)
    last_24h_fc = s_naive(s, h=24, m=24)
    hist = df.iloc[-3 * 24:]
    fig5 = px.area(hist, x=hist.index, y="y", title="Forecast 24h vs. Historie")
    fig5.add_trace(go.Scatter(x=last_24h_fc.index, y=last_24h_fc.values, name="naive_24h", mode="lines"))
    fig5.add_trace(go.Scatter(x=last_week_fc.index, y=last_week_fc.values, name="naive_168h", mode="lines"))

    c1, c2, c3 = st.columns([2, 2, 1])
    st.session_state.setdefault("filter_forecast", False)
    st.session_state.setdefault("refit", False)
    params_path, spec_path = resolve_params_spec()

    with c1:
        with st.popover("📄 Model Card & Run-Metadata"):
            md = model_card_markdown(model_card_meta(kpis=None, gain=None))
            st.markdown(md)
            st.download_button("⬇️ Model Card (Markdown)", md.encode("utf-8"), file_name="model_card.md", mime="text/markdown")

        if st.button("Schnell-Forecast"):
            with st.spinner("Forecast läuft..."):
                st.session_state["filter_forecast"] = True
                st.session_state["refit"] = False
                try:
                    spec = json.load(open(spec_path, "r", encoding="utf-8"))
                    yhat, pi = forecast_from_params(s, H=24, win_days=spec["win_days"], params_path=params_path, spec_path=spec_path)
                    st.session_state["yhat"], st.session_state["pi"] = to_local(yhat, pi)
                except Exception as e:
                    st.warning(f"Initialmodell nicht ladbar: {e}")

        if st.session_state["filter_forecast"]:
            spec = json.load(open(spec_path, "r", encoding="utf-8"))
            fig5.add_trace(go.Scatter(x=st.session_state["yhat"].index, y=st.session_state["yhat"].values, name="SARIMA (init)", mode="lines"))
            fig5.add_trace(go.Scatter(x=st.session_state["pi"].index, y=st.session_state["pi"]["lo"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig5.add_trace(go.Scatter(x=st.session_state["pi"].index, y=st.session_state["pi"]["hi"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(211,211,211,0.35)", name="PI 95% (init)"))
            st.caption(f"Order {spec['order']} | Seasonal Order {spec['seasonal_order']} | Trainingsfenster {spec['win_days']}T | Letzter Refit {spec['last_refit']} ")

    with c2:
        if st.button(f"Refit {refit_days}T"):
            st.session_state["filter_forecast"] = False
            st.session_state["refit"] = True
            with st.spinner(f"Refit {refit_days} & Forecast..."):
                yhat, pi = refit_predict_pi(s, H=24, win_days=refit_days, m=168, alpha=0.05)
                st.session_state["yhat"], st.session_state["pi"] = to_local(yhat, pi)

        if st.session_state["refit"]:
            params_path, spec_path = resolve_params_spec()
            spec = json.load(open(spec_path, "r"))
            fig5.add_trace(go.Scatter(x=st.session_state["yhat"].index, y=st.session_state["yhat"].values, name="SARIMA (refit)", mode="lines"))
            fig5.add_trace(go.Scatter(x=st.session_state["pi"].index, y=st.session_state["pi"]["lo"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig5.add_trace(go.Scatter(x=st.session_state["pi"].index, y=st.session_state["pi"]["hi"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(125,125,125,0.30)", name="PI 95% (refit)"))
            st.caption(f"Order {spec['order']} | Seasonal Order {spec['seasonal_order']} | Trainingsfenster {spec['win_days']}T | Letzter Refit {spec['last_refit']} ")

        if st.button("Feature-Forecast"):
            st.session_state["filter_forecast"] = False
            st.session_state["refit"] = False
            with st.spinner(f"Feature-Modell {refit_days}T & Forecast..."):
                yhat, pi = refit_predict_feature_model(s, H=24, win_days=refit_days)
                st.session_state["feature_yhat"], st.session_state["feature_pi"] = to_local(yhat, pi)

        if "feature_yhat" in st.session_state:
            fig5.add_trace(go.Scatter(x=st.session_state["feature_yhat"].index, y=st.session_state["feature_yhat"].values, name="RF Feature-Modell", mode="lines"))
            fig5.add_trace(go.Scatter(x=st.session_state["feature_pi"].index, y=st.session_state["feature_pi"]["lo"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig5.add_trace(go.Scatter(x=st.session_state["feature_pi"].index, y=st.session_state["feature_pi"]["hi"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(67,170,139,0.22)", name="RF PI 90%"))
            st.caption("Feature-Modell: Random Forest mit Kalender-, Lag- und Rolling-Mean-Features.")

    with c3:
        spec_short = {}
        try:
            spec_short = json.load(open(spec_path, "r", encoding="utf-8"))
        except Exception:
            pass
        kpi_card("Trainingsfenster", float(spec_short.get("win_days", refit_days)), "T", icon="⚙️")

    forecast_start_time = last_24h_fc.index[0]
    fig5.add_shape(type="line", x0=forecast_start_time, y0=0, x1=forecast_start_time, y1=1, yref="paper", line=dict(color="Red", width=2, dash="dash"))
    fig5.add_annotation(x=forecast_start_time, y=1.05, yref="paper", text="Forecast Start", showarrow=False, xanchor="left")
    fig5.data[0].name = "Historie"
    fig5.update_xaxes(title_text="")
    fig5.update_yaxes(title_text="Netzlast (MW)")
    fig5.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig5, width="stretch")

    st.divider()
    st.subheader("Validierung & Modellstatus")
    if st.session_state.get("backtesting", False):
        st.markdown(f"**Modellvergleich** | {eval_days} Tage Validierung")
        comparison_parts = []
        if "df_base" in st.session_state:
            base_cmp = st.session_state["df_base"].reset_index().rename(
                columns={"model": "model", "MAE (MW)": "MAE", "sMAPE (%)": "sMAPE"}
            )
            base_cmp["model_version"] = "baseline_v1"
            comparison_parts.append(base_cmp[["model", "model_version", "MAE", "sMAPE", "MASE_168h"]])
        if "feature_val_df" in st.session_state and not st.session_state["feature_val_df"].empty:
            feature_cmp = st.session_state["feature_val_df"].copy()
            comparison_parts.append(
                feature_cmp[[
                    "model",
                    "model_version",
                    "MAE",
                    "sMAPE",
                    "MASE_168h",
                    "MAE_base",
                    "Gain",
                    "PI_coverage_pct",
                    "PI_mean_width_MW",
                    "folds",
                ]]
            )
        if "last_val_df" in st.session_state and not st.session_state["last_val_df"].empty:
            sarima_cmp = st.session_state["last_val_df"].copy()
            sarima_cmp["model"] = "sarima_exog"
            sarima_cmp["model_version"] = "sarima_exog_v1"
            comparison_parts.append(
                sarima_cmp[[
                    "model",
                    "model_version",
                    "MAE",
                    "sMAPE",
                    "MASE_168h",
                    "MAE_base",
                    "Gain",
                    "PI_coverage_pct",
                    "PI_mean_width_MW",
                    "folds",
                ]]
            )
        if comparison_parts:
            st.dataframe(pd.concat(comparison_parts, ignore_index=True), width="stretch")
        if "feature_val_details" in st.session_state:
            failed = st.session_state["feature_val_details"]
            failed = failed[failed.get("error", "") != ""] if not failed.empty else failed
            if not failed.empty:
                st.warning(f"{len(failed)} Feature-Modell-Fold(s) fehlgeschlagen; Details sind im Ergebnis sichtbar.")
                st.dataframe(failed[["cutoff", "expected_points", "error"]], width="stretch")
    else:
        st.info("Validierung über die Sidebar starten.")

    if "last_val_df" in st.session_state:
        df_sarimax = st.session_state["last_val_df"]
        gain = st.session_state["last_val_gain"]
        meta = st.session_state.get("last_val_meta", {})
    else:
        try:
            df_sarimax, meta, gain = load_validation_json()
        except FileNotFoundError:
            df_sarimax = pd.DataFrame([[1000, 2.5]], columns=["MAE", "sMAPE"])
            gain = 5.0
            meta = {"order": "(1,0,0)", "seasonal_order": "(0,1,0,168)", "train_window_days": 90, "eval_days": 30}

    st.markdown("**Letzte SARIMA-Validierung**")
    show_last_val(df_sarimax, gain)
    st.caption(f"order {meta.get('order')} × {meta.get('seasonal_order')} | {meta.get('train_window_days', '?')}T Train, {meta.get('eval_days', '?')}T Val | {meta.get('validated_at', '(kein Zeitstempel)')}")

    st.session_state.setdefault("qcheck", False)
    params_path, spec_path = resolve_params_spec()
    spec = spec if st.session_state["refit"] else json.load(open(spec_path, "r"))

    if st.button("Qualitätscheck starten"):
        st.session_state["qcheck"] = True
        with st.spinner("Backtesting des aktuellen Modells..."):
            st.caption(f"Trainingsfenster {spec['win_days']}T | Letzter Refit {spec['last_refit']} ")
            kpis, gain = backtest_current_model(s, H=24, eval_days=q_eval_days, win_days=spec["win_days"], m=168, session_res=st.session_state.get("sarima_res"), spec_path=spec_path, params_path=params_path)
            st.session_state["kpis"] = kpis
            st.session_state["gain"] = gain

    if st.session_state.get("qcheck", False):
        c1, c2, c3 = st.columns(3)
        mae = st.session_state["kpis"].get("MAE", np.nan)
        smape_v = st.session_state["kpis"].get("sMAPE", np.nan)
        with c1:
            kpi_card("MAE (MWh)", mae, "MWh", icon="📉")
        with c2:
            kpi_card("sMAPE", smape_v, "%", icon="🧭")
        gate = 5.0
        gain_val = None if (st.session_state["gain"] is None or np.isnan(st.session_state["gain"])) else float(st.session_state["gain"])
        ok = (gain_val is not None) and (gain_val >= gate)
        txt = "Keine gültigen Folds im Fenster." if gain_val is None else f"{gain_val:.1f}% besser als s-Naive(168) – {'OK ✅' if ok else 'unter Gate ❗'}"
        with c3:
            kpi_card("Vorteil ggü. s-Naive", 0.0 if gain_val is None else gain_val, "%", icon=("✅" if ok else "⚠️"), footnote=f"Gate: ≥ {gate:.0f}% · {txt}")


def render_scenarios(s: pd.Series, min_date, max_date) -> None:
    st.subheader("Szenario-Simulation")
    st.caption("What-if auf Historie, getrennt vom Forecast.")
    rng = st.date_input("Historienfenster", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(rng, tuple) and len(rng) == 2:
        start_date, end_date = rng
    else:
        start_date = end_date = rng

    s_loc = s.tz_convert(TIMEZONE)
    y_base = s_loc[(s_loc.index.date >= start_date) & (s_loc.index.date <= end_date)].asfreq("h")
    y_base.name = "History"

    preset = st.selectbox("Preset", ["Manuell", "Feiertag", "Demand Response", "Effizienz", "Wetterstress", "Netto-Last"])
    col1, col2 = st.columns(2)
    with col1:
        use_hw = st.checkbox("Feiertag/Wochenende skalieren", value=preset == "Feiertag")
        use_shift = st.checkbox("Peak -> Off-Peak verschieben", value=preset == "Demand Response")
        use_temp = st.checkbox("Temperatur-Sensitivität", value=preset == "Wetterstress")
    with col2:
        use_eff = st.checkbox("Effizienz-Trend", value=preset == "Effizienz")
        use_event = st.checkbox("Event-Tag(e) skalieren")
        use_net = st.checkbox("Netto-Last PV/Wind", value=preset == "Netto-Last")

    y_scn = y_base.copy()
    holidays_in_range = set()
    if use_hw:
        hol_mult = st.slider("Feiertags-Multiplikator", 0.6, 1.2, 0.88 if preset == "Feiertag" else 0.9, 0.01)
        we_mult = st.slider("Wochenend-Multiplikator", 0.6, 1.2, 0.94 if preset == "Feiertag" else 0.95, 0.01)
        years = range(int(y_scn.index.year.min()), int(y_scn.index.year.max()) + 1)
        holidays_in_range = set(Germany(years=years).keys()) & set(y_scn.index.date)
        y_scn = mult_holiday_weekend(y_scn, holidays_in_range, hol_mult, we_mult)

    if use_shift:
        frac = st.slider("Verschiebe-Anteil", 0.0, 0.5, 0.12 if preset == "Demand Response" else 0.1, 0.01)
        src = st.multiselect("Peak-Stunden (Quelle)", list(range(24)), default=[18, 19, 20, 21])
        dst = st.multiselect("Off-Peak-Stunden (Ziel)", list(range(24)), default=[2, 3, 4, 5])
        y_scn = shift_load(y_scn, frac, src, dst)

    if use_temp:
        delta = st.slider("Delta Temperatur vs. Referenz (C)", -10, 10, -5 if preset == "Wetterstress" else 0)
        kpc = st.slider("Sensitivität (%/°C)", 0.0, 5.0, 1.5, 0.1) / 100.0
        y_scn = temp_adjust(y_scn, delta, kpc, mode="multiplicative")

    if use_eff:
        start = st.date_input("Effizienz ab", value=pd.to_datetime(y_scn.index.min()).date())
        rate = st.slider("Jährliche Änderung (%)", -10.0, 10.0, -2.0 if preset == "Effizienz" else -1.0, 0.1) / 100.0
        y_scn = efficiency_trend(y_scn, start, rate)

    if use_event:
        dates = st.text_input("Event-Tage (YYYY-MM-DD, komma-getrennt)", "")
        mult = st.slider("Event-Multiplikator", 0.5, 1.5, 0.9, 0.01)
        dlist = [d.strip() for d in dates.split(",") if d.strip()]
        if dlist:
            y_scn = event_days(y_scn, dlist, mult)

    if use_net:
        pv_mw = st.slider("PV-Abzug MW", 0, 20000, 6000 if preset == "Netto-Last" else 0, 250)
        wind_mw = st.slider("Wind-Abzug MW", 0, 20000, 4000 if preset == "Netto-Last" else 0, 250)
        y_scn = apply_net_load(y_scn, pv_mw=pv_mw, wind_mw=wind_mw)

    c1, c2 = st.columns(2)
    with c1:
        kpi_card("Delta Peak", float(y_scn.max() - y_base.max()), unit="MW", icon="⚡", footnote="vs. Historie")
    with c2:
        kpi_card("Delta Energie", float((y_scn - y_base).sum()), unit="MWh", icon="📉", footnote="1h-Auflösung")

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=y_base.index, y=y_base, mode="lines", name="Historie"))
    fig6.add_trace(go.Scatter(x=y_scn.index, y=y_scn, mode="lines", name="Simulation"))
    if use_hw:
        for d in holidays_in_range:
            x0 = pd.Timestamp(d, tz=TIMEZONE)
            fig6.add_vrect(x0=x0, x1=x0 + pd.Timedelta(days=1), fillcolor="green", opacity=0.2, line_width=0, layer="below")
    fig6.update_layout(margin=dict(l=40, r=20, t=30, b=20), xaxis_title="", yaxis_title="Netzlast (MW)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig6, width="stretch")


def render_ops() -> None:
    st.subheader("Ops / Monitoring")

    if not Path(METRICS_CSV).exists():
        st.info("Noch keine Metriken vorhanden. CI-Job 'metrics_job.py' täglich ausführen, um zu befüllen.")
        return

    m = pd.read_csv(METRICS_CSV, parse_dates=["forecast_issue", "scored_at"]).sort_values("forecast_issue")
    if m.empty:
        st.info("metrics.csv ist vorhanden, enthält aber noch keine Zeilen.")
        return

    for col in ["MAE", "sMAPE", "Gain", "coverage_pct", "points_compared", "PI_coverage_pct", "PI_mean_width_MW"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")

    latest = m.tail(1).iloc[0]
    valid = m.dropna(subset=["MAE", "sMAPE"])
    ma_7d = valid.tail(7).mean(numeric_only=True) if not valid.empty else pd.Series(dtype=float)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card_value("Letzter Forecast", str(latest.get("forecast_issue", "n/a")), icon="🕑")
    with c2:
        kpi_card("MAE 7d", float(ma_7d.get("MAE", np.nan)), "MW", icon="📉")
    with c3:
        kpi_card("Gain 7d", float(ma_7d.get("Gain", np.nan)), "%", icon="✅")
    with c4:
        kpi_card_value("Coverage zuletzt", f"{float(latest.get('coverage_pct', np.nan)):.1f}%" if pd.notna(latest.get("coverage_pct", np.nan)) else "n/a", icon="🔎")

    st.dataframe(m.tail(14), width="stretch")

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.area(m, x="forecast_issue", y="sMAPE")
        fig.update_traces(line=dict(color="#133046", width=2))
        ma_7d_smape = m["sMAPE"].rolling(7, min_periods=3).mean()
        ma_7d_smape.index = m["forecast_issue"]
        fig.add_trace(go.Scatter(x=ma_7d_smape.index, y=ma_7d_smape.values, mode="lines", line=dict(color="white", width=2, dash="solid"), name="7d MA sMAPE"))
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.area(m, x="forecast_issue", y="Gain", color_discrete_sequence=["#133046"])
        ma_7d = m["Gain"].rolling(7, min_periods=3).mean()
        ma_7d.index = m["forecast_issue"]
        fig.add_trace(go.Scatter(x=ma_7d.index, y=ma_7d.values, mode="lines", line=dict(color="white", width=2, dash="solid"), name="7d MA Gain"))
        st.plotly_chart(fig, width="stretch")

    with col3:
        last = m.dropna(subset=["Gain"]).tail(1)
        if last.empty:
            st.info("Kein 'Gain' vorhanden (noch keine Bewertung).")
            return
        gain = float(m["Gain"].rolling(7, min_periods=3).mean().dropna().tail(1).iloc[0]) if m["Gain"].rolling(7, min_periods=3).mean().notna().any() else float(last["Gain"].iloc[0])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gain,
            number={"suffix": "%"},
            delta={"reference": 0},
            gauge={
                "axis": {"range": [-10, 30]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [-10, 5], "color": "#f94144"},
                    {"range": [5, 10], "color": "#f9c74f"},
                    {"range": [10, 30], "color": "#43aa8b"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "value": gain},
            },
            title={"text": "Gain MA 7d vs. s_naive"},
        ))
        st.plotly_chart(gauge, width="stretch")
        if gain < 5:
            st.error(f"🔴 Gain {gain:.1f}% — unter Gate (≥5%)")
        elif gain < 10:
            st.warning(f"🟡 Gain {gain:.1f}% — Zwischenbereich (5–10%)")
        else:
            st.success(f"🟢 Gain {gain:.1f}% — ok (≥10%)")

    interval_cols = [c for c in ["PI_coverage_pct", "PI_mean_width_MW", "PI_calibration_error_pct"] if c in m.columns]
    if interval_cols:
        st.markdown("**Intervalle**")
        st.dataframe(m[["forecast_issue", *interval_cols]].tail(14), width="stretch")

    forecast_dir = Path("artifacts/forecasts")
    if forecast_dir.exists():
        files = sorted(forecast_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
        st.markdown("**Forecast-Artefakte**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "file": f.name,
                        "modified": pd.Timestamp.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                        "size_kb": round(f.stat().st_size / 1024, 1),
                    }
                    for f in files
                ]
            ),
            width="stretch",
        )


def main() -> None:
    st.set_page_config(layout="wide")
    s = load_app_data(years=2)
    df = prepare_eda_frame(s)

    st.sidebar.title("Stromlast DE")
    page = st.sidebar.radio("Bereich", ["Forecast", "Ops", "EDA", "Szenarien", "Home"], index=0)

    st.title("Stromverbrauch Deutschland")

    if page == "Home":
        render_home()
    elif page == "EDA":
        show_holidays, compare_prev, start_date, end_date, df_view, df_prev, prev_info = read_sidebar_filters(df)
        render_eda(df_view, df_prev, start_date, end_date, prev_info, show_holidays, compare_prev)
    elif page == "Forecast":
        render_forecast(s, df)
    elif page == "Ops":
        render_ops()
    elif page == "Szenarien":
        render_scenarios(s, df.index.date.min(), df.index.date.max())


if __name__ == "__main__":
    main()
