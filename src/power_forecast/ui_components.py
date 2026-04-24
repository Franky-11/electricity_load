import numpy as np
import pandas as pd
import streamlit as st

from .data_quality import calculate_data_quality


KPI_STYLE = """
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
"""


def inject_kpi_style() -> None:
    st.markdown(KPI_STYLE, unsafe_allow_html=True)


def format_thinspace(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ")


def kpi_card(title: str, value: float, unit: str = "", icon: str = "⚡", footnote: str | None = None) -> None:
    inject_kpi_style()
    cls = "pos" if value > 0 else "neg" if value < 0 else "neu"
    sign_icon = "🔺" if value > 0 else "🔻" if value < 0 else "⏸️"
    icon = icon or sign_icon
    val = format_thinspace(value)
    st.markdown(
        f"""
        <div class="kpi {cls}">
          <div class="kpi-head"><span class="kpi-ic">{icon}</span>{title}</div>
          <div class="kpi-val">{val}<span class="kpi-unit"> {unit}</span></div>
          {f'<div class="kpi-foot">{footnote}</div>' if footnote else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card_value(title: str, value, unit: str = "", icon: str = "⚡", footnote: str | None = None) -> None:
    inject_kpi_style()
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-head"><span class="kpi-ic">{icon}</span>{title}</div>
          <div class="kpi-val">{value}<span class="kpi-unit"> {unit}</span></div>
          {f'<div class="kpi-foot">{footnote}</div>' if footnote else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_from_frame(df: pd.DataFrame, column: str, default: float = np.nan) -> float:
    if not isinstance(df, pd.DataFrame) or df.empty or column not in df:
        return default
    return float(df[column].iloc[0])


def show_last_val(df: pd.DataFrame, gain: float | None) -> None:
    mae = _metric_from_frame(df, "MAE")
    smape = _metric_from_frame(df, "sMAPE")
    gate = 5.0
    ok = (gain is not None) and (gain >= gate)
    txt = "Keine gültigen Folds im Fenster."
    if gain is not None:
        txt = f"{gain:.1f}% besser als s-Naive(168) – {'OK ✅' if ok else 'unter Gate ❗'}"

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("MAE (MWh)", mae, "MWh", icon="📉")
    with c2:
        kpi_card("sMAPE", smape, "%", icon="")
    with c3:
        kpi_card(
            "Vorteil ggü. s-Naive",
            0.0 if gain is None else gain,
            "%",
            icon=("✅" if ok else "⚠️"),
            footnote=f"Gate: ≥ {gate:.0f}% · {txt}",
        )


def show_data_quality(s: pd.Series, tz: str = "Europe/Berlin", last: int = 90) -> None:
    quality = calculate_data_quality(s, tz=tz, last=last)

    st.subheader("🧪 Data Quality & Freshness")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card_value("Letzter Datenpunkt", quality.last_local.strftime("%Y-%m-%d %H:%M"), icon="⚡")
    with c2:
        kpi_card_value("Frische-Lag", f"{quality.lag_hours:.1f} h", icon="🕑")
    with c3:
        kpi_card_value(
            f"Missing ({quality.window_days}T)",
            f"{quality.missing_last} / {quality.expected_last} ({quality.coverage_last_pct:.1f}%)",
            icon="🔎",
        )
    with c4:
        kpi_card_value("Duplikate (ges.)", quality.duplicates_total, icon="‼️")

    st.caption(f"TZ: {quality.timezone} | Frequenz: stündlich | DST wird automatisch berücksichtigt")
