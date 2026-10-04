"""Shared presentation components for the retention intelligence workspace."""

from html import escape
from pathlib import Path

import streamlit as st


COLORS = {
    "teal": "#0f766e",
    "blue": "#5470d8",
    "red": "#d45d68",
    "green": "#198577",
    "orange": "#ba8426",
    "ink": "#172b3a",
}
CHURN_COLORS = {"Retained": COLORS["teal"], "Churned": COLORS["red"]}
RISK_COLORS = {
    "Low Risk": COLORS["green"],
    "Medium Risk": COLORS["orange"],
    "High Risk": COLORS["red"],
}
RISK_ORDER = list(RISK_COLORS)


def setup_page():
    st.set_page_config(
        page_title="Retention IQ · Telecom Intelligence",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="auto",
    )
    css = Path(__file__).with_name("assets").joinpath("dashboard.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(eyebrow, title, description, badge="IBM SkillsBuild · Analytics"):
    st.markdown(
        f'<header class="page-header"><div class="page-eyebrow">'
        f'<span>{escape(eyebrow)}</span><span class="context-badge">{escape(badge)}</span>'
        f'</div><h1>{escape(title)}</h1><p>{escape(description)}</p></header>',
        unsafe_allow_html=True,
    )


def kpi_card(label, value, color="blue", suffix="", note=""):
    tone = color if color in COLORS else "teal"
    st.markdown(
        f'<div class="metric-card tone-{tone}">'
        f'<div class="metric-label"><span class="metric-dot"></span>{escape(label)}</div>'
        f'<div class="metric-value">{escape(str(value))}{escape(suffix)}</div>'
        f'<div class="metric-note">{escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(title, description=None):
    st.markdown(f'<h2 class="section-header">{escape(title)}</h2>', unsafe_allow_html=True)
    if description:
        st.caption(description)


def insight_card(text, index=None, tone="teal"):
    marker = f"{index:02d}" if index is not None else "↗"
    st.markdown(
        f'<div class="insight-card tone-{escape(tone)}"><span class="insight-number">'
        f'{marker}</span><p>{escape(text)}</p></div>',
        unsafe_allow_html=True,
    )


def chart(fig, key=None, height=320):
    """Apply one accessible, responsive Plotly style across every page."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12, color="#617384"),
        title_font=dict(size=15, color=COLORS["ink"]),
        margin=dict(l=12, r=18, t=25, b=35),
        colorway=list(COLORS.values())[:6],
        hoverlabel=dict(bgcolor="white", font_size=13, font_color=COLORS["ink"]),
        legend=dict(title_text="", orientation="h", yanchor="top", y=-0.2, x=0),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color="#7b8b98", activecolor=COLORS["teal"]),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=True, title_standoff=14)
    fig.update_yaxes(gridcolor="#edf1f4", zeroline=False, automargin=True, title_standoff=12)
    st.plotly_chart(
        fig, use_container_width=True, theme=None, key=key,
        config={"displaylogo": False, "scrollZoom": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def navigate_to(page):
    st.session_state["workspace_navigation"] = page


def sidebar_brand():
    st.sidebar.markdown(
        '<div class="sidebar-brand"><div class="brand-mark" aria-hidden="true">'
        '<svg width="25" height="25" viewBox="0 0 24 24" fill="none">'
        '<path d="M3 14h4l3-8 4 13 3-8h4" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg></div>'
        '<div><div class="brand-name">Retention<span>IQ</span></div>'
        '<div class="brand-subtitle">TELECOM INTELLIGENCE</div></div></div>'
        '<div class="workspace-label">WORKSPACE</div>',
        unsafe_allow_html=True,
    )


def sidebar_status(model_name, auc, customer_count):
    st.sidebar.markdown(
        '<div class="model-status"><div class="status-label"><span></span> MODEL READY</div>'
        f'<strong>{escape(model_name)}</strong><p>ROC-AUC <b>{auc:.3f}</b>'
        f' <span>·</span> {customer_count:,} records</p></div>',
        unsafe_allow_html=True,
    )
