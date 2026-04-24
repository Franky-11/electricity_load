import pandas as pd
import plotly.graph_objects as go

from .config import TIMEZONE


def eda_trend_chart(df_view: pd.DataFrame, show_holidays: bool = True) -> go.Figure:
    rm24 = df_view["y"].rolling(24, min_periods=1).mean()
    rm168 = df_view["y"].rolling(168, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view["y"], mode="lines", name="Netzlast", line=dict(width=1, color="grey")))
    fig.add_trace(go.Scatter(x=df_view.index, y=rm24, mode="lines", name="RM 24h", line=dict(width=2, color="#4cc9f0")))
    fig.add_trace(go.Scatter(x=df_view.index, y=rm168, mode="lines", name="RM 7d", line=dict(width=2, dash="dot", color="#f9c74f")))

    hol_days = sorted(set(df_view.index.date[df_view["is_hol"]])) if show_holidays else []
    for d in hol_days:
        x0 = pd.Timestamp(d, tz=TIMEZONE)
        fig.add_vrect(x0=x0, x1=x0 + pd.Timedelta(days=1), fillcolor="#43aa8b", opacity=0.18, line_width=0, layer="below")

    fig.update_layout(title="Netzlast mit Rolling Means", height=380, hovermode="x unified", margin=dict(l=40, r=20, t=60, b=30))
    fig.update_yaxes(title_text="Netzlast (MW)")
    fig.update_xaxes(title_text="")
    return fig


def eda_week_profile_chart(df_view: pd.DataFrame, df_prev: pd.DataFrame, compare_prev: bool) -> go.Figure:
    prof = df_view.dropna(subset=["y"]).groupby("h_w")["y"].mean().reindex(range(168))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prof.index, y=prof.values, mode="lines", name="Aktuell"))
    if compare_prev and not df_prev.empty:
        prof_prev = df_prev.dropna(subset=["y"]).groupby("h_w")["y"].mean().reindex(range(168))
        fig.add_trace(go.Scatter(x=prof_prev.index, y=prof_prev.values, mode="lines", name="Vorjahr", line=dict(dash="dash")))
    fig.update_layout(title="Stunde der Woche", height=320, margin=dict(l=40, r=20, t=60, b=30), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_yaxes(title_text="Netzlast (MW)")
    fig.update_xaxes(title_text="Stunde 0..167")
    return fig


def eda_weekday_chart(df_view: pd.DataFrame, df_prev: pd.DataFrame, compare_prev: bool) -> go.Figure:
    dow_mean = df_view.dropna(subset=["y"]).groupby("dow")["y"].mean().reindex(range(7))
    dow_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dow_names, y=dow_mean.values, name="Aktuell", offsetgroup=0, marker_color="#577590"))
    if compare_prev and not df_prev.empty:
        dow_prev = df_prev.dropna(subset=["y"]).groupby("dow")["y"].mean().reindex(range(7))
        fig.add_trace(go.Bar(x=dow_names, y=dow_prev.values, name="Vorjahr", offsetgroup=1, marker_color="#f9c74f"))
    fig.update_layout(title="Wochentag", barmode="group", height=320, margin=dict(l=40, r=20, t=60, b=30))
    fig.update_yaxes(title_text="Netzlast (MW)")
    fig.update_xaxes(title_text="")
    return fig


def eda_month_chart(df_view: pd.DataFrame, df_prev: pd.DataFrame, compare_prev: bool) -> go.Figure:
    month_mean = df_view.groupby("month")["y"].mean().reindex(range(1, 13))
    month_names = ["Jan", "Feb", "Mrz", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    xmonths = list(range(1, 13))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xmonths, y=month_mean.values, fill="tozeroy", name="Aktuell", mode="lines", line=dict(color="#4cc9f0")))
    if compare_prev and not df_prev.empty:
        month_prev = df_prev.groupby("month")["y"].mean().reindex(range(1, 13))
        fig.add_trace(go.Scatter(x=xmonths, y=month_prev.values, name="Vorjahr", mode="lines", line=dict(dash="dash", color="#f9c74f")))
    fig.update_xaxes(tickmode="array", tickvals=xmonths, ticktext=month_names, title_text="")
    fig.update_layout(title="Monat", height=320, margin=dict(l=40, r=20, t=60, b=30))
    fig.update_yaxes(title_text="Netzlast (MW)")
    return fig
