import hashlib
import html
import json
import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv

from ai_service import chat_completion
from analyst_engine import (
    MAX_UNDO_STEPS,
    apply_operation,
    compare_quality,
    detect_anomalies,
    forecast_series,
    join_frames,
    mean_confidence_interval,
    operation_record,
    recipe_json,
    replay_recipe,
    segment_rows,
    statistical_test,
)


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
PROJECTS_DIR = Path(os.getenv("ZUBE_PROJECTS_DIR", "projects"))
ACCENT = "#7C5CFC"
TEAL = "#24C8A5"
PLOTLY_COLORS = [ACCENT, "#A78BFA", TEAL, "#60A5FA", "#F59E0B", "#FB7185"]
WORKSPACE_SECTIONS = [
    "Overview",
    "Prepare",
    "Quality",
    "Visuals",
    "Relationships",
    "KPIs",
    "Decision Lab",
    "AI Brief",
    "Data & Export",
]

st.set_page_config(
    page_title="ZubeAnalystOS · AI Data Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
        :root { --accent:#7C5CFC; --teal:#24C8A5; --ink:#172033; --muted:#667085; }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
        h1, h2, h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-0.035em; }
        .stApp { background:linear-gradient(140deg,#F8F9FF 0%,#FFFFFF 42%,#F4FBFA 100%); color:#344054; }
        [data-testid="stMain"] { min-width:0; overflow-x:hidden; }
        [data-testid="stMain"] :is(h1,h2,h3) { color:#172033; }
        [data-testid="stMain"] h1 { overflow-wrap:anywhere; }
        [data-testid="stSidebar"] { background:#111827; border-right:1px solid rgba(255,255,255,.08); }
        [data-testid="stSidebar"] :is(p, span, label, h1, h2, h3, small) { color:#F9FAFB; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div { background:#1F2937; }
        [data-testid="stSidebar"] [role="radiogroup"] { gap:.35rem; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
          min-height:2.65rem;padding:.48rem .65rem;border:1px solid transparent;border-radius:11px;
          background:rgba(255,255,255,.025);transition:background .16s ease,border-color .16s ease,transform .16s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
          background:rgba(124,92,252,.16);border-color:rgba(167,139,250,.28);transform:translateX(2px);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
          background:linear-gradient(100deg,rgba(101,71,232,.92),rgba(124,92,252,.92));
          border-color:rgba(255,255,255,.22);box-shadow:0 7px 20px rgba(76,52,190,.24);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label :is(p,span) { color:#D7DCEC !important;font-weight:600; }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) :is(p,span) { color:#FFFFFF !important; }
        [data-testid="stSidebar"] [role="radiogroup"] label:focus-within {
          outline:2px solid #24C8A5;outline-offset:2px;
        }
        [data-testid="stSidebar"] .stButton > button {
          background:linear-gradient(100deg,#6547E8 0%,#7C5CFC 55%,#8B6CFF 100%);
          border:1px solid rgba(255,255,255,.2);color:#FFFFFF !important;
          box-shadow:0 8px 22px rgba(109,79,234,.28);transition:transform .18s ease,box-shadow .18s ease,filter .18s ease;
        }
        [data-testid="stSidebar"] .stButton > button :is(p,span,svg) { color:#FFFFFF !important;fill:currentColor; }
        [data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {
          background:linear-gradient(100deg,#7659F0 0%,#8D71FF 55%,#9A82FF 100%);
          border-color:rgba(255,255,255,.42);box-shadow:0 12px 30px rgba(124,92,252,.42);transform:translateY(-1px);
        }
        [data-testid="stSidebar"] .stButton > button:active:not(:disabled) { transform:translateY(0);filter:brightness(.94); }
        [data-testid="stSidebar"] .stButton > button:focus-visible {
          outline:3px solid #24C8A5;outline-offset:3px;box-shadow:0 0 0 5px rgba(36,200,165,.18);
        }
        [data-testid="stSidebar"] .stButton > button:disabled {
          background:#273349;border-color:#3A4861;color:#9CA9BD !important;box-shadow:none;cursor:not-allowed;
        }
        [data-testid="stSidebar"] .stButton > button:disabled :is(p,span,svg) { color:#9CA9BD !important; }
        html { scroll-padding-top:6rem; }
        [data-testid="stHeader"] { background:rgba(255,255,255,.94); backdrop-filter:blur(14px); }
        .block-container { max-width:1500px; padding-top:6.5rem; padding-bottom:5rem; }
        .brand { display:flex; align-items:center; gap:.7rem; margin-bottom:1.5rem; }
        .brand-mark { width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#7C5CFC,#24C8A5);
          display:grid;place-items:center;color:white;font-weight:800;box-shadow:0 8px 24px rgba(124,92,252,.3); }
        .brand-name { color:white;font-family:Manrope;font-weight:800;font-size:1.05rem; }
        .brand-sub { color:#9CA3AF;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase; }
        .product-signature { display:flex;align-items:center;gap:.75rem;margin-bottom:2.4rem; }
        .product-signature .brand-mark { width:42px;height:42px; }
        .product-name { color:#172033;font-family:Manrope;font-size:1.08rem;font-weight:800;letter-spacing:-.025em; }
        .product-tagline { color:#667085;font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase; }
        .eyebrow { color:#7C5CFC;font-weight:700;font-size:.78rem;letter-spacing:.12em;text-transform:uppercase; }
        .hero-title { font-family:Manrope;font-size:clamp(2.1rem,4vw,4.2rem);line-height:1.02;font-weight:800;
          letter-spacing:-.06em;color:#172033;margin:.45rem 0 1rem;max-width:900px; }
        .gradient-text { background:linear-gradient(100deg,#6547E8,#18A98C);-webkit-background-clip:text;color:transparent; }
        .hero-copy { color:#667085;font-size:1.08rem;line-height:1.7;max-width:740px;margin-bottom:2rem; }
        .metric-card { background:rgba(255,255,255,.9);border:1px solid #EAECF0;border-radius:18px;padding:1.05rem 1.15rem;
          min-height:112px;box-shadow:0 8px 30px rgba(16,24,40,.045); }
        .metric-label { color:#667085;font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em; }
        .metric-value { color:#172033;font-family:Manrope;font-size:1.72rem;font-weight:800;margin:.25rem 0 .1rem; }
        .metric-note { color:#98A2B3;font-size:.78rem; }
        .section-kicker { color:#7C5CFC;font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em; }
        .section-title { color:#172033;font-family:Manrope;font-size:1.45rem;font-weight:800;margin:.12rem 0 .15rem; }
        .section-copy { color:#667085;font-size:.9rem;margin-bottom:1rem; }
        .insight-card { background:linear-gradient(135deg,#171F32,#202A42);padding:1.25rem;border-radius:18px;color:#F9FAFB;
          min-height:145px;border:1px solid rgba(255,255,255,.08); }
        .insight-card .label { color:#A78BFA;font-weight:700;font-size:.75rem;text-transform:uppercase;letter-spacing:.08em; }
        .insight-card .value { font-family:Manrope;font-size:1.15rem;font-weight:700;margin-top:.5rem;line-height:1.45; }
        .quality-good { color:#079455;font-weight:700; } .quality-warn { color:#DC6803;font-weight:700; }
        div[data-testid="stPlotlyChart"] { width:100%;min-width:0;background:#FFF;border:1px solid #EAECF0;border-radius:18px;padding:.4rem;
          box-shadow:0 8px 30px rgba(16,24,40,.035); }
        div[data-testid="stPlotlyChart"] > div, div[data-testid="stPlotlyChart"] .js-plotly-plot { width:100% !important;max-width:100%; }
        div[data-testid="stDataFrame"] { max-width:100%;border:1px solid #EAECF0;border-radius:14px;overflow:auto; }
        .stTabs { min-width:0;max-width:100%; }
        .stTabs [data-baseweb="tab-list"] { gap:.3rem;background:#F2F4F7;padding:.35rem;border-radius:14px;overflow-x:auto;scrollbar-width:none; }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display:none; }
        [data-testid="stMain"] .stTabs [data-baseweb="tab"],
        [data-testid="stMain"] .stTabs [role="tab"] {
          flex:0 0 auto;border-radius:10px;padding:.55rem 1rem;font-weight:600;color:#344054 !important;
          background:#E7EAF1 !important;opacity:1 !important;white-space:nowrap;
        }
        [data-testid="stMain"] .stTabs [role="tab"] * { color:#344054 !important;opacity:1 !important; }
        [data-testid="stMain"] .stTabs [role="tab"]:hover { background:#DDDFF0 !important;color:#5138C7 !important; }
        [data-testid="stMain"] .stTabs [role="tab"]:hover * { color:#5138C7 !important; }
        [data-testid="stMain"] .stTabs [aria-selected="true"] {
          background:#FFFFFF !important;border:1px solid #D8D2FF;box-shadow:0 3px 10px rgba(16,24,40,.1);color:#6D4FEA !important;
        }
        [data-testid="stMain"] .stTabs [aria-selected="true"] * { color:#6D4FEA !important;opacity:1 !important; }
        [data-testid="stMain"] .stTabs [role="tab"]:focus-visible { outline:2px solid #24C8A5;outline-offset:2px; }
        .stTabs button[aria-label*="scroll" i] { background:#FFFFFF !important;color:#344054 !important;border:1px solid #D0D5DD; }
        .stTabs button[aria-label*="scroll" i] svg { color:#344054 !important;fill:currentColor; }
        [data-testid="stSegmentedControl"] { max-width:100%;overflow-x:auto;scrollbar-width:none; }
        [data-testid="stSegmentedControl"]::-webkit-scrollbar { display:none; }
        [data-testid="stSegmentedControl"] > div { min-width:max-content; }
        .stButton > button, .stDownloadButton > button { border-radius:12px;font-weight:700;min-height:2.8rem; }
        .stButton > button[kind="primary"] { background:linear-gradient(100deg,#6D4FEA,#7C5CFC);border:0; }
        [data-testid="stFileUploader"] { background:rgba(255,255,255,.88);border:1px solid #E5E7EB;border-radius:22px;
          padding:1.1rem 1.2rem;box-shadow:0 20px 60px rgba(30,41,59,.07); }
        [data-testid="stFileUploaderDropzone"] { border-radius:14px;border:1.5px dashed #C7D0E0;background:#F8FAFC; }
        @media (max-width:1100px) and (min-width:769px) {
          .block-container { padding-left:1.5rem;padding-right:1.5rem; }
          [data-testid="stHorizontalBlock"] { flex-wrap:wrap;gap:1rem; }
          [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 240px;min-width:240px; }
        }
        @media (max-width:768px) {
          html { scroll-padding-top:4rem; }
          [data-testid="stHeader"] { height:3.5rem; }
          .block-container { width:100%;max-width:100%;padding:4.1rem 1rem 3rem;overflow-x:hidden; }
          [data-testid="stSidebar"] { width:min(82vw,300px) !important;max-width:300px; }
          [data-testid="stSidebarContent"] { width:100%;padding-left:.85rem;padding-right:.85rem; }
          [data-testid="stMain"] h1 { font-size:1.7rem !important;line-height:1.15 !important;letter-spacing:-.04em;margin:.25rem 0 .5rem; }
          [data-testid="stMain"] h2 { font-size:1.35rem !important;line-height:1.2 !important; }
          [data-testid="stMain"] h3 { font-size:1.15rem !important; }
          .hero-title { font-size:2.25rem;line-height:1.04;letter-spacing:-.05em; }
          .hero-copy { font-size:.98rem;line-height:1.55;margin-bottom:1.35rem; }
          .product-signature { margin-bottom:1.6rem; }
          .eyebrow { font-size:.7rem;letter-spacing:.1em; }
          [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important;gap:.75rem !important; }
          [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 100% !important;width:100% !important;min-width:0 !important; }
          .metric-card { min-height:auto;padding:1rem;border-radius:16px; }
          .metric-value { font-size:1.55rem; }
          .insight-card { min-height:auto;padding:1rem; }
          .section-title { font-size:1.25rem;line-height:1.25; }
          .section-copy { font-size:.85rem;line-height:1.45; }
          .stTabs [data-baseweb="tab-list"] { border-radius:12px;padding:.3rem;scroll-snap-type:x proximity; }
          .stTabs [data-baseweb="tab"] { min-height:2.7rem;padding:.48rem .85rem;scroll-snap-align:start; }
          [data-testid="stSegmentedControl"] label { min-height:2.65rem;white-space:nowrap; }
          .stButton > button, .stDownloadButton > button { width:100%;min-height:2.85rem; }
          [data-testid="stFileUploader"] { padding:.8rem;border-radius:16px; }
          [data-testid="stFileUploaderDropzone"] { padding:.8rem; }
          div[data-testid="stPlotlyChart"] { padding:.15rem;border-radius:14px; }
          div[data-testid="stDataFrame"] { width:100%;max-width:calc(100vw - 2rem); }
          [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
          [data-testid="stTextArea"] textarea { font-size:16px; }
        }
        @media (max-width:420px) {
          .block-container { padding-left:.75rem;padding-right:.75rem; }
          .hero-title { font-size:2rem; }
          .brand-name { font-size:.98rem; }
          .stTabs [data-baseweb="tab"] { padding:.45rem .72rem; }
          div[data-testid="stDataFrame"] { max-width:calc(100vw - 1.5rem); }
        }
        @media (prefers-reduced-motion:reduce) { * { scroll-behavior:auto !important;transition:none !important; } }
        #MainMenu, footer { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with readable, unique column names."""
    cleaned = df.copy()
    names, seen = [], {}
    for raw in cleaned.columns:
        base = "_".join(str(raw).strip().lower().replace("-", " ").split()) or "column"
        seen[base] = seen.get(base, 0) + 1
        names.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    cleaned.columns = names
    return cleaned


def infer_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Conservatively identify date-like text columns without corrupting IDs."""
    result = df.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        name_hint = any(token in column.lower() for token in ("date", "time", "month", "year"))
        sample = result[column].dropna().astype(str).head(200)
        if name_hint and not sample.empty:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


@st.cache_data(show_spinner=False)
def read_dataset(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    buffer = BytesIO(file_bytes)
    if file_name.lower().endswith((".xlsx", ".xls")):
        frame = pd.read_excel(buffer)
    else:
        try:
            frame = pd.read_csv(buffer)
        except UnicodeDecodeError:
            buffer.seek(0)
            frame = pd.read_csv(buffer, encoding="latin-1")
    return infer_datetime_columns(clean_column_names(frame))


def column_groups(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    dates = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical = [column for column in df.columns if column not in numeric + dates]
    return numeric, categorical, dates


def compact_number(value: float) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= size:
            return f"{value / size:.1f}{unit}"
    return f"{value:,.0f}"


def data_health(df: pd.DataFrame) -> tuple[int, dict]:
    total_cells = max(df.size, 1)
    completeness = 1 - (df.isna().sum().sum() / total_cells)
    uniqueness = 1 - (df.duplicated().sum() / max(len(df), 1))
    constant_columns = sum(df[column].nunique(dropna=True) <= 1 for column in df.columns)
    usefulness = 1 - (constant_columns / max(len(df.columns), 1))
    score = round(100 * (0.55 * completeness + 0.30 * uniqueness + 0.15 * usefulness))
    return max(0, min(score, 100)), {
        "Completeness": completeness,
        "Row uniqueness": uniqueness,
        "Column usefulness": usefulness,
    }


def initialize_workspace(dataset_id: str, source_df: pd.DataFrame) -> None:
    if st.session_state.get("workspace_id") != dataset_id:
        st.session_state.workspace_id = dataset_id
        st.session_state.workspace_section = "Overview"
        st.session_state.working_df = source_df.copy()
        st.session_state.undo_stack = []
        st.session_state.transform_history = []
        st.session_state.kpi_definitions = []


def commit_operation(operation: dict, companion_df: pd.DataFrame | None = None) -> None:
    current = st.session_state.working_df
    if operation["kind"] == "join":
        if companion_df is None:
            raise ValueError("The companion dataset is missing.")
        params = operation["params"]
        updated = join_frames(current, companion_df, params["left_on"], params["right_on"], params["how"])
    else:
        updated = apply_operation(current, operation)
    snapshots = st.session_state.undo_stack
    snapshots.append(current.copy())
    st.session_state.undo_stack = snapshots[-MAX_UNDO_STEPS:]
    st.session_state.working_df = updated
    st.session_state.transform_history.append(operation)


def apply_from_ui(kind: str, label: str, params: dict, companion_df: pd.DataFrame | None = None) -> None:
    operation = operation_record(kind, label, params)
    try:
        commit_operation(operation, companion_df)
    except Exception as exc:
        st.error(f"Transformation could not be applied: {exc}")
        return
    st.toast(f"Applied: {label}", icon="✅")
    st.rerun()


def undo_last_operation() -> None:
    if not st.session_state.undo_stack:
        return
    st.session_state.working_df = st.session_state.undo_stack.pop()
    if st.session_state.transform_history:
        st.session_state.transform_history.pop()
    st.rerun()


def reset_workspace(source_df: pd.DataFrame) -> None:
    st.session_state.working_df = source_df.copy()
    st.session_state.undo_stack = []
    st.session_state.transform_history = []
    st.session_state.kpi_definitions = []
    st.rerun()


def style_figure(fig: go.Figure, title: str | None = None, height: int = 390) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, family="Manrope"), x=0.02) if title else None,
        height=height,
        margin=dict(l=34, r=28, t=60 if title else 28, b=34),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#475467"),
        colorway=PLOTLY_COLORS,
        hoverlabel=dict(bgcolor="#172033", font_color="white", font_family="DM Sans"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor="#EAECF0", title_font=dict(size=12))
    fig.update_yaxes(gridcolor="#F2F4F7", zeroline=False, title_font=dict(size=12))
    return fig


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div><div class="metric-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="section-kicker">{html.escape(kicker)}</div><div class="section-title">{html.escape(title)}</div>'
        f'<div class="section-copy">{html.escape(copy)}</div>', unsafe_allow_html=True,
    )


def dataset_summary(df: pd.DataFrame) -> dict:
    numeric, categorical, dates = column_groups(df)
    summary = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "column_types": {"numeric": numeric, "categorical": categorical, "datetime": dates},
        "data_quality": {
            "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v},
            "duplicate_rows": int(df.duplicated().sum()),
        },
    }
    if numeric:
        summary["numeric_statistics"] = json.loads(df[numeric].describe().round(3).to_json())
        correlations = df[numeric].corr().where(lambda x: ~np.eye(len(x), dtype=bool)).stack()
        if not correlations.empty:
            strongest = correlations.abs().nlargest(8).index
            summary["strongest_correlations"] = [
                {"columns": list(pair), "correlation": round(float(correlations[pair]), 3)} for pair in strongest[::2]
            ]
    category_profiles = {}
    for column in categorical[:12]:
        category_profiles[column] = {
            "unique": int(df[column].nunique(dropna=True)),
            "top_values": {str(k): int(v) for k, v in df[column].value_counts().head(5).items()},
        }
    summary["categorical_profiles"] = category_profiles
    return summary


def call_openrouter_ai(summary: dict, business_context: str, audience: str, decision: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Add OPENROUTER_API_KEY to your .env file to generate AI insights.")
    prompt = f"""
You are the decision-intelligence adviser to a busy, non-technical executive. Turn the dataset profile below
into a practical business brief that helps the reader decide what to do, what not to do, and what to investigate.

Reader: {audience}
Decision or business goal: {decision or 'Not supplied. Identify the most decision-relevant implications, but do not invent a goal.'}
Business context: {business_context or 'Not supplied. Be explicit about what cannot be concluded without it.'}

Dataset profile:
{json.dumps(summary, default=str)[:45000]}

Use these exact Markdown sections:
## Bottom line
In 3-5 plain-English sentences, explain what matters, why it matters to the organization, and whether the
evidence is reliable enough to act on. Lead with the decision implication—not the dataset size.

## Decisions to make now
Give a Markdown table with: Priority | Decision | Why it matters | Expected business effect | Suggested owner.
Include only decisions supported by the available evidence. Expected effect may be qualitative; never invent money,
percentages, or outcomes. Data repair can be a decision, but translate it into its operational or financial risk.

## What the business is telling us
Explain 3-5 useful signals in everyday language. For each signal, use this pattern:
**Signal — business meaning — sensible response.** Refer to coded field names only when necessary and never pretend
to know what an abbreviation means without a supplied definition.

## Risks before acting
Explain how data quality or missing context could cause a bad decision. State clearly what is observed versus inferred.
Turn technical issues into consequences (for example: duplicate records can make demand or risk look larger and make
a predictive tool appear more reliable than it is).

## 30-day action plan
Give 3-5 sequenced actions with an owner role and a tangible deliverable. Recommend statistical modeling only when it
directly supports the stated decision; modeling is not itself business value.

## Questions that unlock a stronger decision
Ask no more than four short questions. Each question must say what decision its answer would improve.

## Technical appendix
Preserve useful analytical detail here for an analyst: sample size, distributions, correlations, outliers, assumptions,
and recommended analytical treatment.

Rules:
- Write for an intelligent leader who does not know statistics. Use short sentences and concrete language.
- Outside the Technical appendix, do not use notation such as r, SD, quartiles, p-values, IQR, or model jargon.
- Do not lead with means, correlations, or column counts. Translate evidence into impact first.
- Do not diagnose patients, claim causality, guess the dataset's source, or claim a coded value is good/bad without definitions.
- If the context is insufficient for a commercial or operational recommendation, say so plainly and propose the fastest
  way to obtain the missing information.
- Use specific values sparingly when they help a decision. Keep the executive portion under 700 words.
"""
    return chat_completion(
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
        system_prompt="You are rigorous, commercially fluent, plain-spoken, and allergic to unsupported claims.",
        user_prompt=prompt,
        temperature=0.25,
        max_tokens=1600,
    )


def call_openrouter_question(df: pd.DataFrame, question: str) -> tuple[str, str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Add OPENROUTER_API_KEY to your .env file to ask questions about the data.")
    row_limit = 300
    if len(df) <= row_limit:
        evidence = df
        scope = f"all {len(df):,} rows"
    else:
        evidence = df.sample(row_limit, random_state=42)
        scope = f"a reproducible sample of {row_limit:,} from {len(df):,} rows"
    csv_evidence = evidence.to_csv(index=False)[:65000]
    prompt = f"""
Answer the business user's question using the row-level evidence and dataset profile below.

Question: {question}
Evidence scope: {scope}
Dataset profile: {json.dumps(dataset_summary(df), default=str)[:25000]}
Row-level evidence (CSV):
{csv_evidence}

Give a direct plain-English answer first. Then provide:
- Evidence: the exact values, counts, groups, or time periods that support the answer.
- Business meaning: why the answer may matter.
- Confidence and limits: clearly state that the answer used {scope}; never imply sampled rows represent an exact full-data calculation.
- Suggested next step: one practical follow-up.

Do not invent calculations, definitions, causality, or domain context. If the supplied evidence cannot answer the question,
say exactly what is missing. Keep the answer under 500 words and avoid statistical jargon unless the user asks for it.
"""
    answer = chat_completion(
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
        system_prompt="You answer questions from supplied data evidence with precision and transparent limits.",
        user_prompt=prompt,
        temperature=0.1,
        max_tokens=1200,
    )
    return answer, scope


def build_markdown_report(df: pd.DataFrame, file_name: str, ai_report: str | None) -> str:
    health, factors = data_health(df)
    numeric, categorical, dates = column_groups(df)
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    lines = [
        "# ZubeAnalystOS Dataset Report", "", f"**Source:** {file_name}",
        f"**Generated:** {datetime.now().strftime('%d %b %Y, %H:%M')}", "",
        "## Dataset overview", "", f"- Rows: {len(df):,}", f"- Columns: {len(df.columns):,}",
        f"- Data health score: {health}/100", f"- Duplicate rows: {df.duplicated().sum():,}",
        f"- Numeric / categorical / date fields: {len(numeric)} / {len(categorical)} / {len(dates)}", "",
        "## Quality factors", "",
    ]
    lines.extend(f"- {name}: {value:.1%}" for name, value in factors.items())
    lines.extend(["", "## Missing values", ""])
    lines.extend([f"- {column}: {count:,} ({count / len(df):.1%})" for column, count in missing.items()] or ["No missing values detected."])
    if numeric:
        lines.extend(["", "## Numeric summary", "", df[numeric].describe().round(2).to_markdown(), ""])
    if ai_report:
        lines.extend(["", "## AI decision brief", "", ai_report, ""])
    return "\n".join(lines)


def render_sidebar(df: pd.DataFrame, file_name: str, file_size: int) -> str:
    numeric, categorical, dates = column_groups(df)
    st.sidebar.markdown(
        '<div class="brand"><div class="brand-mark">Z</div><div><div class="brand-name">ZubeAnalystOS</div>'
        '<div class="brand-sub">Decision intelligence</div></div></div>', unsafe_allow_html=True,
    )
    st.sidebar.caption("WORKSPACE MENU")
    selected_section = st.sidebar.radio(
        "Workspace",
        WORKSPACE_SECTIONS,
        key="workspace_section",
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption("ACTIVE DATASET")
    st.sidebar.markdown(f"**{file_name}**")
    st.sidebar.caption(f"{file_size / 1024:.1f} KB · {len(df):,} rows · {len(df.columns)} fields")
    st.sidebar.divider()
    st.sidebar.caption("FIELD INVENTORY")
    c1, c2, c3 = st.sidebar.columns(3)
    c1.metric("Numeric", len(numeric))
    c2.metric("Text", len(categorical))
    c3.metric("Dates", len(dates))
    st.sidebar.divider()
    st.sidebar.caption("SESSION")
    st.sidebar.info("Your dataset stays in this Streamlit session and is only summarized when you request AI analysis.")
    return selected_section


def render_landing() -> None:
    st.markdown(
        '<div class="product-signature"><div class="brand-mark">Z</div><div>'
        '<div class="product-name">ZubeAnalystOS</div><div class="product-tagline">Decision intelligence workspace</div>'
        '</div></div>', unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">AI-assisted business intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">From raw spreadsheet to <span class="gradient-text">clear decisions.</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-copy">Upload a dataset and explore its shape, quality, relationships, and business signals in minutes—then turn the evidence into an executive-ready brief.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Drop a CSV or Excel workbook here",
        type=["csv", "xlsx", "xls"],
        help="CSV, XLSX, and XLS files are supported.",
        key="landing_upload",
    )
    st.markdown("### Built for the first useful answer")
    a, b, c = st.columns(3)
    with a:
        metric_card("Explore", "Interactive", "Filterable charts with rich hover context")
    with b:
        metric_card("Diagnose", "Quality-aware", "Missingness, duplicates, outliers, and types")
    with c:
        metric_card("Decide", "AI-assisted", "Evidence-grounded executive recommendations")
    return uploaded


def render_overview(df: pd.DataFrame) -> None:
    numeric, categorical, dates = column_groups(df)
    health, _ = data_health(df)
    missing_pct = df.isna().sum().sum() / max(df.size, 1)
    duplicates = int(df.duplicated().sum())
    cols = st.columns(5)
    cards = [
        ("Rows", compact_number(len(df)), "observations loaded"),
        ("Columns", str(len(df.columns)), f"{len(numeric)} numeric fields"),
        ("Data health", f"{health}/100", "weighted quality score"),
        ("Missing", f"{missing_pct:.1%}", f"{int(df.isna().sum().sum()):,} empty cells"),
        ("Duplicates", compact_number(duplicates), f"{duplicates / max(len(df), 1):.1%} of rows"),
    ]
    for col, card in zip(cols, cards):
        with col:
            metric_card(*card)

    st.write("")
    section_header("Dataset pulse", "What stands out at first glance", "A fast structural read before deeper exploration.")
    left, middle, right = st.columns(3)
    with left:
        completeness = 1 - missing_pct
        message = f"{completeness:.1%} of all cells are populated."
        st.markdown(f'<div class="insight-card"><div class="label">Completeness</div><div class="value">{message}</div></div>', unsafe_allow_html=True)
    with middle:
        widest = max((df[c].nunique(dropna=True), c) for c in df.columns)[1]
        st.markdown(f'<div class="insight-card"><div class="label">Highest cardinality</div><div class="value">{html.escape(widest)} has {df[widest].nunique(dropna=True):,} distinct values.</div></div>', unsafe_allow_html=True)
    with right:
        type_message = f"{len(numeric)} numeric, {len(categorical)} categorical, and {len(dates)} date fields detected."
        st.markdown(f'<div class="insight-card"><div class="label">Analysis mix</div><div class="value">{type_message}</div></div>', unsafe_allow_html=True)

    st.write("")
    chart_left, chart_right = st.columns([1.15, 1])
    with chart_left:
        if numeric:
            chosen = st.selectbox("Distribution metric", numeric, key="overview_metric")
            fig = px.histogram(df, x=chosen, marginal="box", color_discrete_sequence=[ACCENT], opacity=.88)
            st.plotly_chart(style_figure(fig, f"Distribution of {chosen}"), width="stretch", config={"displaylogo": False})
        else:
            st.info("Add numeric fields to unlock distribution analysis.")
    with chart_right:
        type_counts = pd.DataFrame({
            "Type": ["Numeric", "Categorical", "Datetime"],
            "Fields": [len(numeric), len(categorical), len(dates)],
        })
        fig = px.pie(type_counts, names="Type", values="Fields", hole=.65, color_discrete_sequence=PLOTLY_COLORS)
        fig.update_traces(textposition="outside", textinfo="label+value", hovertemplate="%{label}: %{value} fields<extra></extra>")
        fig.add_annotation(text=f"<b>{len(df.columns)}</b><br><span style='font-size:11px'>FIELDS</span>", showarrow=False)
        st.plotly_chart(style_figure(fig, "Field composition"), width="stretch", config={"displaylogo": False})


def render_quality(df: pd.DataFrame) -> None:
    score, factors = data_health(df)
    section_header("Quality lab", "Know where the data can mislead you", "Completeness, duplication, consistency, and potential outliers at field level.")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "/100", "font": {"size": 42}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0}, "bar": {"color": TEAL if score >= 80 else "#F59E0B"},
            "bgcolor": "#F2F4F7", "borderwidth": 0,
            "steps": [{"range": [0, 60], "color": "#FEF3F2"}, {"range": [60, 80], "color": "#FFFAEB"}, {"range": [80, 100], "color": "#ECFDF3"}],
        },
    ))
    left, right = st.columns([.8, 1.2])
    with left:
        st.plotly_chart(style_figure(gauge, "Data health score", 320), width="stretch", config={"displayModeBar": False})
    with right:
        factor_df = pd.DataFrame({"Factor": factors.keys(), "Score": [v * 100 for v in factors.values()]})
        fig = px.bar(factor_df, x="Score", y="Factor", orientation="h", text_auto=".1f", color="Score",
                     color_continuous_scale=[[0, "#FB7185"], [.65, "#F59E0B"], [1, TEAL]], range_color=[0, 100])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(range=[0, 100], ticksuffix="%")
        st.plotly_chart(style_figure(fig, "Quality factors", 320), width="stretch", config={"displaylogo": False})

    missing = df.isna().sum().sort_values(ascending=True)
    missing = missing[missing > 0]
    q1, q2 = st.columns(2)
    with q1:
        if not missing.empty:
            miss_df = pd.DataFrame({"Column": missing.index, "Missing": missing.values})
            miss_df["Percent"] = miss_df["Missing"] / len(df) * 100
            fig = px.bar(miss_df.tail(18), x="Percent", y="Column", orientation="h", text_auto=".1f", color_discrete_sequence=["#FB7185"])
            fig.update_xaxes(ticksuffix="%")
            st.plotly_chart(style_figure(fig, "Missing data by field"), width="stretch", config={"displaylogo": False})
        else:
            st.success("Complete dataset — no missing cells detected.")
    with q2:
        numeric, _, _ = column_groups(df)
        outliers = []
        for column in numeric:
            series = df[column].dropna()
            if len(series) < 4:
                continue
            q1v, q3v = series.quantile([.25, .75])
            iqr = q3v - q1v
            count = int(((series < q1v - 1.5 * iqr) | (series > q3v + 1.5 * iqr)).sum()) if iqr else 0
            outliers.append({"Column": column, "Potential outliers": count, "Rate": count / len(series) * 100})
        if outliers:
            outlier_df = pd.DataFrame(outliers).sort_values("Rate").tail(18)
            fig = px.bar(outlier_df, x="Rate", y="Column", orientation="h", text_auto=".1f", color_discrete_sequence=["#F59E0B"])
            fig.update_xaxes(ticksuffix="%")
            st.plotly_chart(style_figure(fig, "Potential outliers · IQR method"), width="stretch", config={"displaylogo": False})
        else:
            st.info("Numeric fields are required for outlier screening.")

    profile = pd.DataFrame({
        "Field": df.columns,
        "Type": [str(dtype) for dtype in df.dtypes],
        "Populated": [int(df[c].notna().sum()) for c in df.columns],
        "Missing %": [(df[c].isna().mean() * 100) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        "Example": [str(df[c].dropna().iloc[0])[:60] if df[c].notna().any() else "—" for c in df.columns],
    })
    st.dataframe(profile, width="stretch", hide_index=True, column_config={"Missing %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})


def render_relationships(df: pd.DataFrame) -> None:
    numeric, categorical, _ = column_groups(df)
    section_header("Relationship map", "Find variables that move together", "Correlation is a signal for investigation—not proof of causation.")
    if len(numeric) < 2:
        st.info("At least two numeric fields are needed for relationship analysis.")
        return
    selected = st.multiselect("Fields in correlation matrix", numeric, default=numeric[:min(10, len(numeric))], max_selections=16)
    if len(selected) >= 2:
        corr = df[selected].corr()
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale=["#2563EB", "#F8FAFC", "#E54868"], aspect="auto")
        fig.update_layout(coloraxis_colorbar=dict(title="r"))
        st.plotly_chart(style_figure(fig, "Correlation matrix", max(430, 38 * len(selected))), width="stretch", config={"displaylogo": False})
    xcol, ycol, colorcol = st.columns(3)
    with xcol:
        x = st.selectbox("X axis", numeric, index=0, key="rel_x")
    with ycol:
        y = st.selectbox("Y axis", numeric, index=min(1, len(numeric) - 1), key="rel_y")
    with colorcol:
        color_options = ["None"] + [c for c in categorical if df[c].nunique(dropna=True) <= 20]
        color = st.selectbox("Segment", color_options, key="rel_color")
    plot_df = df.sample(min(len(df), 5000), random_state=42) if len(df) > 5000 else df
    fig = px.scatter(plot_df, x=x, y=y, color=None if color == "None" else color, opacity=.68,
                     color_discrete_sequence=PLOTLY_COLORS, render_mode="webgl", hover_data=plot_df.columns[:min(4, len(plot_df.columns))])
    st.plotly_chart(style_figure(fig, f"{y} vs {x}", 470), width="stretch", config={"displaylogo": False})
    if len(df) > 5000:
        st.caption("Scatter plot uses a reproducible 5,000-row sample for responsive exploration.")


def summarize_chart_data(
    df: pd.DataFrame,
    category: str,
    measure: str,
    aggregation: str,
    top_n: int = 15,
    series: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Create a compact, chart-ready table without changing the working dataset."""
    grouping = [category] + ([series] if series else [])
    working = df[grouping + ([] if measure == "Record count" else [measure])].copy()
    for column in grouping:
        working[column] = working[column].fillna("(Missing)").astype(str)
    if measure == "Record count":
        result = working.groupby(grouping, dropna=False).size().reset_index(name="Records")
        value_column = "Records"
    else:
        result = working.groupby(grouping, dropna=False)[measure].agg(aggregation.lower()).reset_index()
        value_column = measure
    totals = result.groupby(category, dropna=False)[value_column].sum().nlargest(top_n).index
    result = result[result[category].isin(totals)]
    return result, value_column


def render_visual_explorer(df: pd.DataFrame) -> None:
    numeric, categorical, dates = column_groups(df)
    section_header(
        "Visual studio",
        "Turn the dataset into the right story",
        "Choose a chart family and ZubeAnalystOS will show only the controls that chart needs.",
    )
    low_cardinality = [column for column in categorical if df[column].nunique(dropna=True) <= 40]
    categories = low_cardinality or categorical
    choices: list[str] = []
    if categorical:
        choices.extend(["Column chart", "Bar chart", "Pie chart", "Doughnut chart", "Funnel chart"])
    if numeric and categorical:
        choices.extend(["Grouped / stacked columns", "Treemap", "Sunburst", "Radar chart", "Waterfall chart"])
    if dates and numeric:
        choices.extend(["Line chart", "Area chart"])
    if len(numeric) >= 2:
        choices.extend(["Scatter plot", "Bubble chart", "Correlation heatmap"])
    if numeric:
        choices.extend(["Histogram", "Box plot", "Violin plot"])
    if not choices:
        st.info("This dataset has no chartable numeric, category, or date fields.")
        return

    chooser, note = st.columns([1.2, 2])
    with chooser:
        chart_type = st.selectbox("Chart type", choices, key="visual_chart_type")
    with note:
        st.caption("Interactive: hover for detail, zoom or select points, and use the camera icon to export a high-resolution PNG.")

    aggregation_options = ["Sum", "Mean", "Median", "Min", "Max"]
    fig: go.Figure
    title = chart_type

    if chart_type in {"Column chart", "Bar chart", "Pie chart", "Doughnut chart", "Funnel chart"}:
        controls = st.columns(4)
        category = controls[0].selectbox("Category", categorical, key="basic_category")
        measure = controls[1].selectbox("Value", ["Record count"] + numeric, key="basic_measure")
        aggregation = controls[2].selectbox("Calculation", aggregation_options, disabled=measure == "Record count", key="basic_aggregation")
        top_n = controls[3].slider("Categories", 3, 30, 12, key="basic_top_n")
        chart_df, value_column = summarize_chart_data(df, category, measure, aggregation, top_n)
        chart_df = chart_df.sort_values(value_column, ascending=False)
        title = f"{value_column} by {category}"
        if chart_type == "Column chart":
            fig = px.bar(chart_df, x=category, y=value_column, text_auto=".3s", color_discrete_sequence=[ACCENT])
        elif chart_type == "Bar chart":
            fig = px.bar(chart_df.sort_values(value_column), x=value_column, y=category, orientation="h", text_auto=".3s", color_discrete_sequence=[TEAL])
        elif chart_type in {"Pie chart", "Doughnut chart"}:
            if (chart_df[value_column] < 0).any():
                st.info("Pie and doughnut charts require non-negative values. Choose Record count or another measure.")
                return
            fig = px.pie(chart_df, names=category, values=value_column, hole=.58 if chart_type == "Doughnut chart" else 0, color_discrete_sequence=PLOTLY_COLORS)
            fig.update_traces(textposition="inside", textinfo="percent+label")
        else:
            fig = px.funnel(chart_df, x=value_column, y=category, color=category, color_discrete_sequence=PLOTLY_COLORS)

    elif chart_type == "Grouped / stacked columns":
        controls = st.columns(5)
        category = controls[0].selectbox("Category", categories, key="stack_category")
        series_options = [column for column in categories if column != category]
        if not series_options:
            st.info("This chart needs two category fields with a manageable number of values.")
            return
        series = controls[1].selectbox("Series", series_options, key="stack_series")
        measure = controls[2].selectbox("Value", ["Record count"] + numeric, key="stack_measure")
        aggregation = controls[3].selectbox("Calculation", aggregation_options, disabled=measure == "Record count", key="stack_aggregation")
        mode = controls[4].selectbox("Layout", ["Grouped", "Stacked", "100% stacked"], key="stack_mode")
        chart_df, value_column = summarize_chart_data(df, category, measure, aggregation, 15, series)
        if mode == "100% stacked":
            totals = chart_df.groupby(category)[value_column].transform("sum").replace(0, np.nan)
            chart_df["Share"] = chart_df[value_column] / totals * 100
            value_column = "Share"
        fig = px.bar(chart_df, x=category, y=value_column, color=series, barmode="group" if mode == "Grouped" else "stack", color_discrete_sequence=PLOTLY_COLORS)
        if mode == "100% stacked":
            fig.update_yaxes(ticksuffix="%")
        title = f"{value_column} by {category} and {series}"

    elif chart_type in {"Line chart", "Area chart"}:
        controls = st.columns(4)
        date = controls[0].selectbox("Date", dates, key="trend_date")
        value = controls[1].selectbox("Measure", numeric, key="trend_measure")
        aggregation = controls[2].selectbox("Calculation", aggregation_options, key="trend_aggregation")
        frequency = controls[3].selectbox("Frequency", ["Day", "Week", "Month", "Quarter", "Year"], index=2, key="trend_frequency")
        rules = {"Day": "D", "Week": "W", "Month": "ME", "Quarter": "QE", "Year": "YE"}
        timeline = df[[date, value]].dropna().set_index(date).resample(rules[frequency])[value].agg(aggregation.lower()).reset_index()
        fig = px.area(timeline, x=date, y=value, color_discrete_sequence=[ACCENT]) if chart_type == "Area chart" else px.line(timeline, x=date, y=value, markers=True, color_discrete_sequence=[ACCENT])
        fig.update_traces(line=dict(width=3))
        title = f"{value} over time"

    elif chart_type in {"Scatter plot", "Bubble chart"}:
        controls = st.columns(4)
        x = controls[0].selectbox("X axis", numeric, key="point_x")
        y_options = [column for column in numeric if column != x] or numeric
        y = controls[1].selectbox("Y axis", y_options, key="point_y")
        color_options = ["None"] + categories
        color = controls[2].selectbox("Color", color_options, key="point_color")
        size = None
        if chart_type == "Bubble chart":
            size_options = [column for column in numeric if column not in {x, y}] or numeric
            size = controls[3].selectbox("Bubble size", size_options, key="point_size")
        plot_df = df.sample(min(len(df), 5000), random_state=42) if len(df) > 5000 else df
        if size:
            plot_df = plot_df.copy()
            plot_df["Bubble size"] = pd.to_numeric(plot_df[size], errors="coerce").abs().fillna(0)
            size = "Bubble size"
        fig = px.scatter(plot_df, x=x, y=y, color=None if color == "None" else color, size=size, size_max=42, opacity=.7, color_discrete_sequence=PLOTLY_COLORS, render_mode="webgl")
        title = f"{y} vs {x}"

    elif chart_type in {"Histogram", "Box plot", "Violin plot"}:
        controls = st.columns(3)
        value = controls[0].selectbox("Measure", numeric, key="distribution_measure")
        segment = controls[1].selectbox("Segment", ["None"] + categories, key="distribution_segment")
        color = None if segment == "None" else segment
        if chart_type == "Histogram":
            bins = controls[2].slider("Bins", 10, 80, 30, key="histogram_bins")
            fig = px.histogram(df, x=value, color=color, nbins=bins, marginal="rug", opacity=.82, color_discrete_sequence=PLOTLY_COLORS)
        elif chart_type == "Box plot":
            fig = px.box(df, x=color, y=value, color=color, points="outliers", color_discrete_sequence=PLOTLY_COLORS)
        else:
            fig = px.violin(df, x=color, y=value, color=color, box=True, points="outliers", color_discrete_sequence=PLOTLY_COLORS)
        title = f"Distribution of {value}"

    elif chart_type == "Correlation heatmap":
        selected = st.multiselect("Measures", numeric, default=numeric[:min(8, len(numeric))], max_selections=16, key="heatmap_measures")
        if len(selected) < 2:
            st.info("Select at least two measures to build a correlation heatmap.")
            return
        corr = df[selected].corr()
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale=["#2563EB", "#F8FAFC", "#E54868"], aspect="auto")
        title = "Correlation heatmap"

    elif chart_type in {"Treemap", "Sunburst"}:
        controls = st.columns([2, 1, 1])
        hierarchy = controls[0].multiselect("Hierarchy", categories, default=categories[:min(2, len(categories))], max_selections=3, key="hierarchy_fields")
        measure = controls[1].selectbox("Value", ["Record count"] + numeric, key="hierarchy_measure")
        aggregation = controls[2].selectbox("Calculation", aggregation_options, disabled=measure == "Record count", key="hierarchy_aggregation")
        if not hierarchy:
            st.info("Select at least one hierarchy field.")
            return
        working = df[hierarchy + ([] if measure == "Record count" else [measure])].copy()
        working[hierarchy] = working[hierarchy].fillna("(Missing)").astype(str)
        if measure == "Record count":
            working["Records"] = 1
            value_column = "Records"
        else:
            value_column = measure
        grouped = working.groupby(hierarchy, dropna=False)[value_column].agg("sum" if measure == "Record count" else aggregation.lower()).reset_index()
        if (grouped[value_column] < 0).any():
            st.info("Treemap and sunburst area values must be non-negative. Choose Record count or another measure.")
            return
        figure_function = px.treemap if chart_type == "Treemap" else px.sunburst
        fig = figure_function(grouped, path=hierarchy, values=value_column, color=hierarchy[0], color_discrete_sequence=PLOTLY_COLORS)
        title = f"{value_column} hierarchy"

    elif chart_type == "Radar chart":
        controls = st.columns(3)
        category = controls[0].selectbox("Category", categories, key="radar_category")
        value = controls[1].selectbox("Measure", numeric, key="radar_measure")
        aggregation = controls[2].selectbox("Calculation", aggregation_options, key="radar_aggregation")
        chart_df, value_column = summarize_chart_data(df, category, value, aggregation, 12)
        chart_df = chart_df.sort_values(value_column, ascending=False)
        fig = px.line_polar(chart_df, r=value_column, theta=category, line_close=True, markers=True, color_discrete_sequence=[ACCENT])
        fig.update_traces(fill="toself")
        title = f"{value_column} profile by {category}"

    else:  # Waterfall chart
        controls = st.columns(3)
        category = controls[0].selectbox("Category", categories, key="waterfall_category")
        value = controls[1].selectbox("Measure", numeric, key="waterfall_measure")
        aggregation = controls[2].selectbox("Calculation", aggregation_options, key="waterfall_aggregation")
        chart_df, value_column = summarize_chart_data(df, category, value, aggregation, 15)
        chart_df = chart_df.sort_values(value_column, key=lambda values: values.abs(), ascending=False)
        fig = go.Figure(go.Waterfall(x=chart_df[category], y=chart_df[value_column], measure=["relative"] * len(chart_df), connector={"line": {"color": "#D0D5DD"}}))
        title = f"Contribution to {value_column} by {category}"

    st.plotly_chart(
        style_figure(fig, title, 540),
        width="stretch",
        config={"displaylogo": False, "responsive": True, "toImageButtonOptions": {"format": "png", "scale": 2}},
    )


def render_data_studio(df: pd.DataFrame, source_df: pd.DataFrame, dataset_id: str) -> None:
    section_header(
        "Data preparation studio",
        "Clean, reshape, and combine—with an undo button",
        "Every change applies to a working copy. The original upload remains untouched and each step is recorded.",
    )
    action_col, undo_col, reset_col, recipe_col = st.columns([1.6, .8, .8, 1])
    action_col.metric("Working dataset", f"{len(df):,} × {len(df.columns):,}", f"{len(st.session_state.transform_history)} transformations")
    with undo_col:
        st.write("")
        if st.button("Undo last", disabled=not st.session_state.undo_stack, width="stretch"):
            undo_last_operation()
    with reset_col:
        st.write("")
        if st.button("Reset all", disabled=not st.session_state.transform_history, width="stretch"):
            reset_workspace(source_df)
    with recipe_col:
        st.write("")
        recipe = recipe_json("ZubeAnalystOS recipe", st.session_state.transform_history, st.session_state.kpi_definitions)
        st.download_button("Export recipe", recipe, "zubeanalystos_recipe.json", "application/json", width="stretch")

    quality = compare_quality(source_df, df)
    with st.expander("Before-and-after quality validation", expanded=bool(st.session_state.transform_history)):
        st.dataframe(quality, hide_index=True, width="stretch")

    clean_tab, wrangle_tab, join_tab, history_tab = st.tabs(["Clean", "Wrangle", "Combine datasets", "History & recipes"])
    numeric, categorical, _ = column_groups(df)

    with clean_tab:
        st.markdown("#### Missing values")
        missing_columns = [column for column in df.columns if df[column].isna().any()]
        if missing_columns:
            with st.form("missing_form"):
                a, b, c = st.columns([1.2, 1, 1])
                column = a.selectbox("Field", missing_columns)
                allowed = ["Drop rows", "Drop column", "Mode", "Custom value"]
                if pd.api.types.is_numeric_dtype(df[column]):
                    allowed[2:2] = ["Mean", "Median"]
                method = b.selectbox("Treatment", allowed)
                value = c.text_input("Custom replacement", help="Used only when treatment is Custom value.")
                if st.form_submit_button("Apply missing-value treatment", type="primary"):
                    if method == "Custom value" and not value.strip():
                        st.error("Enter the replacement value you want to use.")
                    else:
                        apply_from_ui("missing", f"{method} for missing {column}", {"column": column, "method": method, "value": value})
        else:
            st.success("No missing values remain in the working dataset.")

        st.divider()
        st.markdown("#### Duplicate records")
        duplicate_count = int(df.duplicated().sum())
        st.caption(f"{duplicate_count:,} exact duplicate rows currently detected.")
        with st.form("duplicate_form"):
            subset = st.multiselect("Fields that define a duplicate", df.columns.tolist(), help="Leave empty to compare every field.")
            keep_label = st.selectbox("Record to keep", ["First", "Last", "Remove every duplicated record"])
            keep_map = {"First": "first", "Last": "last", "Remove every duplicated record": False}
            if st.form_submit_button("Remove duplicates"):
                if duplicate_count == 0 and not subset:
                    st.info("No exact duplicates were found. Choose business-key fields if you want to check duplicates by selected columns.")
                else:
                    apply_from_ui("deduplicate", "Remove duplicate records", {"subset": subset, "keep": keep_map[keep_label]})

        st.divider()
        type_col, outlier_col = st.columns(2)
        with type_col:
            st.markdown("#### Correct a field type")
            with st.form("type_form"):
                column = st.selectbox("Field", df.columns, key="type_column")
                target = st.selectbox("Convert to", ["Number", "Date/time", "Text", "Category", "Boolean"])
                st.caption("Values that cannot be converted become missing so they can be reviewed safely.")
                if st.form_submit_button("Convert field"):
                    apply_from_ui("convert_type", f"Convert {column} to {target}", {"column": column, "target": target})
        with outlier_col:
            st.markdown("#### Treat potential outliers")
            if numeric:
                with st.form("outlier_form"):
                    column = st.selectbox("Numeric field", numeric, key="outlier_column")
                    method = st.selectbox("Treatment", ["Cap values", "Remove rows", "Replace with missing"])
                    factor = st.slider("IQR sensitivity", 1.0, 3.0, 1.5, .25)
                    if st.form_submit_button("Apply outlier treatment"):
                        apply_from_ui("outliers", f"{method} for {column}", {"column": column, "method": method, "factor": factor})
            else:
                st.info("No numeric fields are available.")

    with wrangle_tab:
        task = st.segmented_control(
            "Wrangling task",
            ["Filter rows", "Calculated field", "Group & summarize", "Pivot", "Reshape"],
            default="Filter rows",
            key="wrangling_task",
        )
        if task == "Filter rows":
            with st.form("filter_form"):
                a, b, c = st.columns(3)
                column = a.selectbox("Field", df.columns, key="filter_column")
                is_numeric = pd.api.types.is_numeric_dtype(df[column])
                operators = ["equals", "not equal", "is missing", "is populated"]
                operators += ["greater than", "less than", "at least", "at most"] if is_numeric else ["contains"]
                operator = b.selectbox("Condition", operators)
                value = c.text_input("Value", help="Not required for missing/populated conditions.")
                if st.form_submit_button("Keep matching rows", type="primary"):
                    if operator not in ("is missing", "is populated") and not value.strip():
                        st.error("Enter a value for this filter condition.")
                    else:
                        apply_from_ui("filter", f"Filter {column} {operator}", {"column": column, "operator": operator, "value": value})
        elif task == "Calculated field":
            with st.form("calculate_form"):
                a, b = st.columns([1, 2])
                name = a.text_input("New field name", placeholder="gross_margin")
                expression = b.text_input("Formula", placeholder="(revenue - cost) / revenue")
                st.caption(f"Available numeric fields: {', '.join(numeric) or 'none'}. Use arithmetic operators only.")
                if st.form_submit_button("Create calculated field", type="primary"):
                    if not name.strip() or not expression.strip():
                        st.error("Enter both a new field name and a calculation formula.")
                    else:
                        apply_from_ui("calculated", f"Create calculated field {name}", {"name": name.strip(), "expression": expression})
        elif task == "Group & summarize":
            with st.form("group_form"):
                a, b, c = st.columns(3)
                groups = a.multiselect("Group by", df.columns, max_selections=3)
                values = b.multiselect("Measures", numeric)
                aggregation = c.selectbox("Calculation", ["sum", "mean", "median", "min", "max", "count"])
                st.warning("This creates a summarized working dataset. Export a recipe or use Undo to return to row-level data.")
                if st.form_submit_button("Create summary", type="primary"):
                    if not groups or not values:
                        st.error("Select at least one grouping field and one measure.")
                    else:
                        apply_from_ui("group", f"Group by {', '.join(groups)}", {"groups": groups, "values": values, "aggregation": aggregation})
        elif task == "Pivot":
            with st.form("pivot_form"):
                a, b, c, d = st.columns(4)
                index = a.multiselect("Row fields", df.columns, max_selections=3)
                columns = b.selectbox("Column field", df.columns)
                values = c.selectbox("Value field", numeric if numeric else df.columns)
                aggregation = d.selectbox("Calculation", ["sum", "mean", "median", "min", "max", "count"])
                if st.form_submit_button("Create pivot table", type="primary"):
                    if not index:
                        st.error("Select at least one row field for the pivot table.")
                    else:
                        apply_from_ui("pivot", "Create pivot table", {"index": index, "columns": columns, "values": values, "aggregation": aggregation})
        else:
            with st.form("melt_form"):
                a, b = st.columns(2)
                id_vars = a.multiselect("Identifier fields", df.columns)
                value_vars = b.multiselect("Fields to stack", df.columns, help="Choose fields that are not also selected as identifiers.")
                c, d = st.columns(2)
                var_name = c.text_input("New measure-name field", "measure")
                value_name = d.text_input("New value field", "value")
                if st.form_submit_button("Reshape to long format", type="primary"):
                    if not value_vars:
                        st.error("Select at least one field to stack.")
                    elif set(id_vars) & set(value_vars):
                        st.error("A field cannot be both an identifier and a field to stack.")
                    elif not var_name.strip() or not value_name.strip():
                        st.error("Enter names for the new measure and value fields.")
                    elif var_name == value_name:
                        st.error("The measure-name field and value field must have different names.")
                    else:
                        apply_from_ui("melt", "Reshape to long format", {"id_vars": id_vars, "value_vars": value_vars, "var_name": var_name.strip(), "value_name": value_name.strip()})

    with join_tab:
        st.markdown("#### Join another dataset")
        companion = st.file_uploader("Upload the table to combine", type=["csv", "xlsx", "xls"], key="join_upload")
        if companion:
            try:
                right_df = read_dataset(companion.getvalue(), companion.name)
                st.caption(f"Companion table: {len(right_df):,} rows × {len(right_df.columns)} fields")
                with st.form("join_form"):
                    a, b, c = st.columns(3)
                    left_on = a.selectbox("Current dataset key", df.columns)
                    right_on = b.selectbox("Companion dataset key", right_df.columns)
                    how_label = c.selectbox("Join method", ["Keep matching rows", "Keep every current row", "Keep every companion row", "Keep all rows"])
                    how_map = {"Keep matching rows": "inner", "Keep every current row": "left", "Keep every companion row": "right", "Keep all rows": "outer"}
                    if st.form_submit_button("Combine datasets", type="primary"):
                        params = {"left_on": left_on, "right_on": right_on, "how": how_map[how_label], "companion": companion.name}
                        apply_from_ui("join", f"{how_label} with {companion.name}", params, right_df)
            except Exception as exc:
                st.error(f"Could not prepare the companion dataset: {exc}")

    with history_tab:
        history = st.session_state.transform_history
        if history:
            history_df = pd.DataFrame([
                {"Step": i + 1, "Transformation": item["label"], "Applied": item["timestamp"]} for i, item in enumerate(history)
            ])
            st.dataframe(history_df, hide_index=True, width="stretch")
        else:
            st.info("No transformations have been applied yet.")
        st.markdown("#### Apply a saved recipe")
        uploaded_recipe = st.file_uploader("Upload a ZubeAnalystOS recipe", type=["json"], key="recipe_upload")
        if uploaded_recipe and st.button("Replay recipe on the original dataset", type="primary"):
            try:
                payload = json.loads(uploaded_recipe.getvalue())
                transformed, skipped = replay_recipe(source_df, payload.get("operations", []))
                st.session_state.undo_stack.append(df.copy())
                st.session_state.working_df = transformed
                st.session_state.transform_history = payload.get("operations", [])
                st.session_state.kpi_definitions = payload.get("kpis", [])
                if skipped:
                    st.warning("Some steps were skipped: " + " | ".join(skipped))
                st.rerun()
            except Exception as exc:
                st.error(f"Recipe could not be loaded: {exc}")
        st.markdown("#### Saved projects")
        st.caption("Projects save transformation steps and KPI definitions—not the uploaded data itself.")
        project_col, save_col = st.columns([2, 1])
        project_name = project_col.text_input("Project name", placeholder="Quarterly sales workspace", label_visibility="collapsed")
        if save_col.button("Save project", width="stretch", disabled=not project_name.strip()):
            safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", project_name.strip()).strip("_")[:60]
            PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
            (PROJECTS_DIR / f"{safe_name}.json").write_text(
                recipe_json(project_name.strip(), st.session_state.transform_history, st.session_state.kpi_definitions),
                encoding="utf-8",
            )
            st.toast(f"Saved project: {project_name}", icon="💾")
            st.rerun()
        project_files = sorted(PROJECTS_DIR.glob("*.json")) if PROJECTS_DIR.exists() else []
        if project_files:
            selected_project = st.selectbox("Open a saved project", project_files, format_func=lambda path: path.stem.replace("_", " ").title())
            if st.button("Apply saved project to this dataset"):
                try:
                    payload = json.loads(selected_project.read_text(encoding="utf-8"))
                    transformed, skipped = replay_recipe(source_df, payload.get("operations", []))
                    st.session_state.undo_stack.append(df.copy())
                    st.session_state.working_df = transformed
                    st.session_state.transform_history = payload.get("operations", [])
                    st.session_state.kpi_definitions = payload.get("kpis", [])
                    if skipped:
                        st.warning("Some dataset-specific steps were skipped: " + " | ".join(skipped))
                    st.rerun()
                except Exception as exc:
                    st.error(f"Saved project could not be applied: {exc}")


def render_kpi_board(df: pd.DataFrame) -> None:
    section_header(
        "KPI board",
        "Define the numbers leadership should watch",
        "Turn dataset fields into reusable business measures with targets and status signals.",
    )
    numeric, _, _ = column_groups(df)
    definitions = st.session_state.kpi_definitions
    if definitions:
        cards = st.columns(min(4, len(definitions)))
        for index, definition in enumerate(definitions):
            column = definition["column"]
            aggregation = definition["aggregation"]
            if column not in df.columns:
                value = np.nan
            elif aggregation == "count":
                value = float(df[column].count())
            else:
                value = float(getattr(df[column], aggregation)())
            target = definition.get("target")
            delta = value - target if target is not None and np.isfinite(value) else None
            with cards[index % len(cards)]:
                st.metric(
                    definition["name"],
                    f"{definition.get('prefix', '')}{value:,.{definition.get('decimals', 1)}f}{definition.get('suffix', '')}" if np.isfinite(value) else "Unavailable",
                    f"{delta:+,.1f} vs target" if delta is not None else None,
                )
        st.caption("KPIs recalculate automatically whenever the working dataset changes.")
    else:
        st.info("Create your first KPI below. KPI definitions are included in exported project recipes.")

    with st.expander("Create a KPI", expanded=not definitions):
        if not numeric:
            st.warning("At least one numeric field is required to define a KPI.")
        else:
            with st.form("kpi_form"):
                a, b, c = st.columns(3)
                name = a.text_input("Business name", placeholder="Total revenue")
                column = b.selectbox("Measure field", numeric)
                aggregation = c.selectbox("Calculation", ["sum", "mean", "median", "min", "max", "count"])
                d, e, f, g = st.columns(4)
                target_enabled = d.checkbox("Compare with target")
                target = e.number_input("Target", value=0.0, help="Used only when Compare with target is selected.")
                prefix = f.text_input("Prefix", placeholder="$")
                suffix = g.text_input("Suffix", placeholder="%")
                if st.form_submit_button("Add KPI", type="primary"):
                    if not name.strip():
                        st.error("Enter a clear business name for this KPI.")
                    elif any(item["name"].lower() == name.strip().lower() for item in definitions):
                        st.error("A KPI with this name already exists.")
                    else:
                        st.session_state.kpi_definitions.append({
                            "name": name.strip(), "column": column, "aggregation": aggregation,
                            "target": target if target_enabled else None, "prefix": prefix, "suffix": suffix, "decimals": 1,
                        })
                        st.rerun()
    if definitions and st.button("Clear KPI board"):
        st.session_state.kpi_definitions = []
        st.rerun()

    if definitions:
        names = [item["name"] for item in definitions]
        chosen = st.selectbox("Trend or segment a KPI", names)
        definition = next(item for item in definitions if item["name"] == chosen)
        category_candidates = [c for c in df.columns if c != definition["column"] and df[c].nunique(dropna=True) <= 30]
        if category_candidates:
            category = st.selectbox("Break down by", category_candidates)
            grouped = df.groupby(category, dropna=False)[definition["column"]].agg(definition["aggregation"]).nlargest(20).sort_values().reset_index()
            fig = px.bar(grouped, x=definition["column"], y=category, orientation="h", color_discrete_sequence=[TEAL], text_auto=".3s")
            st.plotly_chart(style_figure(fig, f"{chosen} by {category}"), width="stretch", config={"displaylogo": False})


def render_advanced_lab(df: pd.DataFrame, dataset_id: str) -> None:
    section_header(
        "Decision science lab",
        "Test assumptions, find unusual records, build segments, and look ahead",
        "These tools support investigation. Results should be reviewed with business and domain knowledge before action.",
    )
    numeric, categorical, dates = column_groups(df)
    stats_tab, anomaly_tab, segment_tab, forecast_tab = st.tabs(["Statistical tests", "Anomalies", "Segments", "Forecast"])

    with stats_tab:
        if not numeric:
            st.info("Numeric fields are required for statistical testing.")
        else:
            group_candidates = [column for column in df.columns if 2 <= df[column].nunique(dropna=True) <= 20]
            test_name = st.selectbox("Question to test", ["Confidence interval for an average", "Two-group mean comparison", "Multi-group mean comparison", "Category association", "Numeric relationship"])
            confidence = st.slider("Confidence level", .80, .99, .95, .01, format="%.0f%%")
            with st.form("stats_form"):
                numeric_column = group_column = second_column = None
                if test_name == "Confidence interval for an average":
                    numeric_column = st.selectbox("Measure", numeric)
                elif test_name in ("Two-group mean comparison", "Multi-group mean comparison"):
                    a, b = st.columns(2)
                    numeric_column = a.selectbox("Measure", numeric)
                    group_column = b.selectbox("Groups", group_candidates or df.columns)
                elif test_name == "Category association":
                    candidates = group_candidates if len(group_candidates) >= 2 else df.columns.tolist()
                    a, b = st.columns(2)
                    group_column = a.selectbox("First category", candidates)
                    second_column = b.selectbox("Second category", candidates, index=min(1, len(candidates) - 1))
                else:
                    a, b = st.columns(2)
                    numeric_column = a.selectbox("First measure", numeric)
                    second_column = b.selectbox("Second measure", numeric, index=min(1, len(numeric) - 1))
                submitted = st.form_submit_button("Run test", type="primary")
            if submitted:
                try:
                    if test_name == "Confidence interval for an average":
                        result = {"name": test_name, **mean_confidence_interval(df[numeric_column], confidence)}
                    else:
                        result = statistical_test(df, test_name, numeric_column, group_column, second_column, confidence)
                    st.session_state[f"stats_result_{dataset_id}"] = result
                except Exception as exc:
                    st.error(str(exc))
            result = st.session_state.get(f"stats_result_{dataset_id}")
            if result:
                if "lower" in result:
                    st.success(f"Estimated average: **{result['mean']:,.3f}**. The {confidence:.0%} confidence range is **{result['lower']:,.3f} to {result['upper']:,.3f}** based on {result['n']:,} values.")
                else:
                    verdict = "The observed difference is unlikely to be random alone." if result["significant"] else "The available evidence does not establish a dependable difference."
                    st.markdown(f"#### {result['name']}")
                    st.info(f"{verdict} Probability under the no-difference assumption: {result['pvalue']:.4f}.")
                    with st.expander("Technical result"):
                        st.json(result)
            if numeric_column:
                with st.expander("Statistical distribution · Seaborn"):
                    figure, axis = plt.subplots(figsize=(10, 4.5))
                    if group_column and group_column in df.columns and df[group_column].nunique(dropna=True) <= 15:
                        sns.violinplot(data=df, x=group_column, y=numeric_column, inner="quart", color="#A78BFA", ax=axis)
                        axis.tick_params(axis="x", rotation=30)
                    else:
                        sns.histplot(data=df, x=numeric_column, kde=True, color=ACCENT, ax=axis)
                    axis.set_title(f"Statistical distribution of {numeric_column}", loc="left", fontweight="bold")
                    sns.despine()
                    figure.tight_layout()
                    st.pyplot(figure, width="stretch")
                    plt.close(figure)

    with anomaly_tab:
        if len(numeric) < 1:
            st.info("Numeric fields are required for anomaly detection.")
        else:
            selected = st.multiselect("Signals used to identify unusual records", numeric, default=numeric[:min(5, len(numeric))], key="anomaly_fields")
            contamination = st.slider("Expected unusual share", .01, .20, .05, .01, format="%.0f%%")
            if st.button("Detect unusual records", type="primary", disabled=not selected):
                try:
                    st.session_state[f"anomaly_result_{dataset_id}"] = detect_anomalies(df, selected, contamination)
                except Exception as exc:
                    st.error(str(exc))
            result = st.session_state.get(f"anomaly_result_{dataset_id}")
            if result is not None:
                flagged = result[result["anomaly"]].sort_values("anomaly_score", ascending=False)
                st.metric("Records flagged for review", f"{len(flagged):,}", f"{len(flagged) / max(len(result), 1):.1%} of analyzable rows")
                st.dataframe(flagged.head(100), hide_index=True, width="stretch")
                a, b = st.columns(2)
                a.download_button("Download flagged records", flagged.to_csv(index=False), "anomaly_review.csv", "text/csv", width="stretch")
                if b.button("Add anomaly flags to working data", width="stretch"):
                    apply_from_ui("anomaly", "Add anomaly review flags", {"columns": selected, "contamination": contamination})

    with segment_tab:
        if len(numeric) < 2:
            st.info("At least two numeric fields are required to create meaningful segments.")
        else:
            selected = st.multiselect("Characteristics used for segmentation", numeric, default=numeric[:min(4, len(numeric))], key="segment_fields")
            clusters = st.slider("Number of segments", 2, 8, 3)
            if st.button("Build segments", type="primary", disabled=len(selected) < 2):
                try:
                    segmented, profiles = segment_rows(df, selected, clusters)
                    st.session_state[f"segment_result_{dataset_id}"] = (segmented, profiles, selected, clusters)
                except Exception as exc:
                    st.error(str(exc))
            result = st.session_state.get(f"segment_result_{dataset_id}")
            if result:
                segmented, profiles, used_fields, used_clusters = result
                st.markdown("#### Segment profiles")
                st.dataframe(profiles, hide_index=True, width="stretch")
                fig = px.parallel_coordinates(profiles, dimensions=used_fields, color="segment", color_continuous_scale="Viridis")
                st.plotly_chart(style_figure(fig, "How the segments differ", 470), width="stretch", config={"displaylogo": False})
                if st.button("Add segment labels to working data"):
                    apply_from_ui("segment", "Add behavioral segments", {"columns": used_fields, "clusters": used_clusters})

    with forecast_tab:
        if not dates or not numeric:
            st.info("A detected date field and a numeric measure are required for forecasting.")
        else:
            with st.form("forecast_form"):
                a, b, c, d = st.columns(4)
                date_column = a.selectbox("Date", dates)
                value_column = b.selectbox("Measure", numeric)
                frequency = c.selectbox("Frequency", ["Day", "Week", "Month", "Quarter"], index=2)
                periods = d.slider("Periods ahead", 2, 24, 6)
                submitted = st.form_submit_button("Create forecast", type="primary")
            if submitted:
                try:
                    st.session_state[f"forecast_result_{dataset_id}"] = forecast_series(df, date_column, value_column, periods, frequency)
                    st.session_state[f"forecast_meta_{dataset_id}"] = (date_column, value_column)
                except Exception as exc:
                    st.error(str(exc))
            result = st.session_state.get(f"forecast_result_{dataset_id}")
            if result:
                history, forecast = result
                date_column, value_column = st.session_state[f"forecast_meta_{dataset_id}"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history[date_column], y=history["value"], name="Actual", line=dict(color=ACCENT, width=3)))
                fig.add_trace(go.Scatter(x=forecast[date_column], y=forecast["upper"], line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=forecast[date_column], y=forecast["lower"], fill="tonexty", fillcolor="rgba(36,200,165,.16)", line=dict(width=0), name="Approx. range"))
                fig.add_trace(go.Scatter(x=forecast[date_column], y=forecast["forecast"], name="Forecast", line=dict(color=TEAL, width=3, dash="dash")))
                st.plotly_chart(style_figure(fig, f"Forecast for {value_column}", 470), width="stretch", config={"displaylogo": False})
                st.caption("Forecasts extend historical patterns; they do not account for future campaigns, policy changes, shocks, or other external drivers.")
                st.download_button("Download forecast", forecast.to_csv(index=False), "forecast.csv", "text/csv")


def render_ai(df: pd.DataFrame, dataset_id: str) -> None:
    section_header("AI decision brief", "A leadership brief, not a statistics lecture", "Tell ZubeAnalystOS who is deciding and what is at stake. The main brief stays in plain English; analytical detail is kept in a separate appendix.")
    profile_col, decision_col = st.columns(2)
    with profile_col:
        audience = st.selectbox(
            "Who will read this?",
            ["Business executive", "Operations leader", "Finance leader", "Sales or marketing leader", "Healthcare executive", "Data or analytics leader"],
        )
    with decision_col:
        decision = st.text_input(
            "What decision should this support?",
            placeholder="Example: Where should we focus resources next quarter?",
        )
    context = st.text_area(
        "Useful business context",
        placeholder="Example: This is patient screening data for a regional hospital. We want to improve early intervention without increasing unnecessary referrals.",
        height=105,
    )
    report_key = f"ai_report_{dataset_id}"
    if st.button("Generate decision brief", type="primary", icon="✨", width="stretch"):
        with st.spinner("Drafting your brief—this usually takes 10–45 seconds…"):
            try:
                st.session_state[report_key] = call_openrouter_ai(dataset_summary(df), context, audience, decision)
            except requests.Timeout:
                st.error("The AI provider took too long, so ZubeAnalystOS stopped waiting after 70 seconds.")
                st.info("Please try again. Your dataset and selections are still available, and no report was lost.")
            except requests.HTTPError as exc:
                detail = exc.response.text[:400] if exc.response is not None else str(exc)
                st.error(f"OpenRouter request failed: {detail}")
            except (requests.RequestException, RuntimeError) as exc:
                st.error(str(exc))
    report = st.session_state.get(report_key)
    if report:
        appendix_marker = "## Technical appendix"
        if appendix_marker in report:
            executive_brief, technical_detail = report.split(appendix_marker, 1)
            st.markdown(executive_brief)
            with st.expander("Technical appendix · for analysts"):
                st.markdown(f"{appendix_marker}{technical_detail}")
        else:
            st.markdown(report)
    else:
        st.info("Your executive brief will appear here and remain available while you explore other tabs.")

    st.divider()
    section_header(
        "Ask the data",
        "Ask a question in everyday language",
        "ZubeAnalystOS answers from row-level evidence, explains the business meaning, and states when sampling limits certainty.",
    )
    question_key = f"question_history_{dataset_id}"
    if question_key not in st.session_state:
        st.session_state[question_key] = []
    with st.form("ask_data_form"):
        question = st.text_input(
            "Your question",
            placeholder="Which region contributes the most revenue, and what appears to be driving it?",
        )
        ask = st.form_submit_button("Ask ZubeAnalystOS", type="primary")
    if ask and question.strip():
        with st.spinner("Reading the row-level evidence—this usually takes under a minute…"):
            try:
                answer, scope = call_openrouter_question(df, question.strip())
                st.session_state[question_key].append({"question": question.strip(), "answer": answer, "scope": scope})
            except requests.Timeout:
                st.error("The AI provider took too long, so ZubeAnalystOS stopped waiting after 70 seconds. Please try again.")
            except requests.HTTPError as exc:
                detail = exc.response.text[:400] if exc.response is not None else str(exc)
                st.error(f"OpenRouter request failed: {detail}")
            except (requests.RequestException, RuntimeError) as exc:
                st.error(str(exc))
    for exchange in reversed(st.session_state[question_key][-5:]):
        with st.chat_message("user"):
            st.markdown(exchange["question"])
        with st.chat_message("assistant"):
            st.markdown(exchange["answer"])
            st.caption(f"Evidence scope: {exchange['scope']}")


def render_data_table(df: pd.DataFrame, file_name: str, ai_report: str | None) -> None:
    section_header("Data workspace", "Inspect and export the evidence", "Search the preview, review statistics, and take the cleaned data or full report with you.")
    query = st.text_input("Search all fields", placeholder="Type a value to filter the preview…", icon="🔎")
    display_df = df
    if query:
        mask = df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        display_df = df[mask]
        st.caption(f"{len(display_df):,} matching rows")
    st.dataframe(display_df, width="stretch", hide_index=True, height=470)
    csv = df.to_csv(index=False).encode("utf-8")
    report = build_markdown_report(df, file_name, ai_report)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download cleaned CSV", csv, "cleaned_dataset.csv", "text/csv", width="stretch")
    c2.download_button("Download analysis report", report, "zubeanalystos_report.md", "text/markdown", width="stretch")
    c3.download_button("Download data dictionary", pd.DataFrame({"field": df.columns, "dtype": df.dtypes.astype(str), "non_null": df.notna().sum(), "unique": df.nunique(dropna=True)}).to_csv(index=False), "data_dictionary.csv", "text/csv", width="stretch")


inject_styles()

if "active_file" not in st.session_state:
    st.session_state.active_file = None

if st.session_state.active_file is None:
    uploaded_file = render_landing()
    if uploaded_file is not None:
        st.session_state.active_file = {
            "name": uploaded_file.name,
            "bytes": uploaded_file.getvalue(),
        }
        st.rerun()
else:
    active = st.session_state.active_file
    try:
        source_df = read_dataset(active["bytes"], active["name"])
    except Exception as exc:
        st.error(f"I couldn't read this dataset: {exc}")
        if st.button("Choose another file"):
            st.session_state.active_file = None
            st.rerun()
        st.stop()
    if source_df.empty or len(source_df.columns) == 0:
        st.error("This file does not contain a usable table.")
        st.stop()

    dataset_id = hashlib.sha256(active["bytes"]).hexdigest()[:12]
    initialize_workspace(dataset_id, source_df)
    df = st.session_state.working_df
    selected_section = render_sidebar(df, active["name"], len(active["bytes"]))
    if st.sidebar.button("↻ Analyze another file", width="stretch"):
        st.session_state.active_file = None
        st.rerun()

    st.markdown(f'<div class="eyebrow">{html.escape(selected_section)} workspace</div>', unsafe_allow_html=True)
    st.title(active["name"])
    st.caption(f"Working copy: {len(df):,} rows × {len(df.columns)} fields · {len(st.session_state.transform_history)} transformations · updated {datetime.now().strftime('%H:%M')}")
    if selected_section == "Overview":
        render_overview(df)
    elif selected_section == "Prepare":
        render_data_studio(df, source_df, dataset_id)
    elif selected_section == "Quality":
        render_quality(df)
    elif selected_section == "Visuals":
        render_visual_explorer(df)
    elif selected_section == "Relationships":
        render_relationships(df)
    elif selected_section == "KPIs":
        render_kpi_board(df)
    elif selected_section == "Decision Lab":
        render_advanced_lab(df, dataset_id)
    elif selected_section == "AI Brief":
        render_ai(df, dataset_id)
    elif selected_section == "Data & Export":
        render_data_table(df, active["name"], st.session_state.get(f"ai_report_{dataset_id}"))
