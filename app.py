"""
Streamlit web app for the Geopolitical Network Analysis project.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from src.knowledge_graph import (
        KnowledgeGraphBuilder,
        KnowledgeGraphEnricher,
        KnowledgeGraphExporter,
    )
    from src.llm import LLMFactory
    HAS_OPTIONAL_BACKEND = True
    BACKEND_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - UI fallback
    HAS_OPTIONAL_BACKEND = False
    BACKEND_IMPORT_ERROR = str(exc)


DATA_DIR = Path("./gdelt_processed_data")
RESULTS_DIR = Path("./results")
VIS_DIR = Path("./gdelt_visualizations")
VIS_DIR.mkdir(parents=True, exist_ok=True)

COUNTRY_COORDS: Dict[str, tuple[float, float]] = {
    "USA": (37.09, -95.71),
    "CAN": (56.13, -106.35),
    "CHN": (35.86, 104.20),
    "JPN": (36.20, 138.25),
    "KOR": (35.91, 127.77),
    "IND": (20.59, 78.96),
    "PAK": (30.38, 69.35),
    "RUS": (61.52, 105.32),
    "UKR": (48.38, 31.17),
    "GBR": (55.38, -3.44),
    "FRA": (46.23, 2.21),
    "DEU": (51.17, 10.45),
    "ISR": (31.05, 34.85),
    "IRN": (32.43, 53.69),
    "TUR": (38.96, 35.24),
    "SAU": (23.89, 45.08),
    "EGY": (26.82, 30.80),
    "AUS": (-25.27, 133.78),
    "BRA": (-14.24, -51.93),
    "ZAF": (-30.56, 22.94),
}

REGION_MAP: Dict[str, str] = {
    "USA": "North America",
    "CAN": "North America",
    "CHN": "East Asia",
    "JPN": "East Asia",
    "KOR": "East Asia",
    "IND": "South Asia",
    "PAK": "South Asia",
    "RUS": "Eurasia",
    "UKR": "Eurasia",
    "GBR": "Europe",
    "FRA": "Europe",
    "DEU": "Europe",
    "ISR": "Middle East",
    "IRN": "Middle East",
    "TUR": "Middle East",
    "SAU": "Middle East",
    "EGY": "Middle East",
    "AUS": "Oceania",
    "BRA": "South America",
    "ZAF": "Africa",
}

REGION_COLORS: Dict[str, str] = {
    "North America": "#2d6cdf",
    "East Asia": "#ed7d31",
    "South Asia": "#d1495b",
    "Eurasia": "#2a9d8f",
    "Europe": "#4caf50",
    "Middle East": "#c89100",
    "Oceania": "#cc5a71",
    "South America": "#3d405b",
    "Africa": "#2b7a78",
}

st.set_page_config(
    page_title="Geopolitical Conflict Pulse",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(211, 73, 91, 0.14), transparent 26%),
                radial-gradient(circle at top right, rgba(45, 108, 223, 0.14), transparent 28%),
                linear-gradient(180deg, #f7f4ec 0%, #eef2f8 100%);
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(29, 78, 216, 0.86));
            color: #f8fafc;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.3rem;
            line-height: 1.1;
        }
        .hero p {
            margin: 0.7rem 0 0 0;
            max-width: 58rem;
            font-size: 1rem;
            color: rgba(248, 250, 252, 0.88);
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        }
        .metric-label {
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #5b6474;
        }
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 0.15rem;
        }
        .metric-caption {
            color: #506072;
            margin-top: 0.35rem;
            font-size: 0.92rem;
        }

        /* Enhanced Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            padding: 0.8rem 0;
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            height: 54px;
            background-color: rgba(255, 255, 255, 0.3);
            border-radius: 12px 12px 0 0;
            padding: 0 32px;
            font-size: 1.15rem;
            font-weight: 700;
            color: #334155;
            border: 1px solid rgba(15, 23, 42, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin-right: 4px;
            text-transform: none;
            box-shadow: none;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(255, 255, 255, 0.8);
            color: #2563eb;
            border-color: rgba(37, 99, 235, 0.3);
            transform: translateY(-1px);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1e293b, #3b82f6) !important;
            color: white !important;
            border-bottom: 4px solid #60a5fa !important;
            box-shadow: 0 10px 25px rgba(30, 41, 59, 0.15) !important;
        }

        .stTabs [data-baseweb="tab-panel"] {
            background: rgba(255, 255, 255, 0.85);
            border-radius: 0 20px 20px 20px;
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 1);
            backdrop-filter: blur(12px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.05);
            margin-top: -1px;
            color: #1e293b !important;
        }

        /* Ensure markdown text is visible on light backgrounds */
        [data-testid="stMarkdownContainer"] p, 
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            color: #1e293b !important;
        }

        /* Premium Button Styling */
        .stButton > button {
            border-radius: 12px !important;
            height: 50px !important;
            font-weight: 600 !important;
            background: linear-gradient(135deg, #2563eb, #1e40af) !important;
            color: white !important;
            border: none !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
            font-size: 1rem !important;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3) !important;
            color: white !important;
        }

        .stButton > button:active {
            transform: translateY(0px) !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-image: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
            border-right: 1px solid rgba(15, 23, 42, 0.08) !important;
        }
        
        [data-testid="stSidebar"] .stSelectbox label, 
        [data-testid="stSidebar"] .stRadio label {
            color: #475569 !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_processed_data() -> Optional[Dict[str, Any]]:
    required = [
        DATA_DIR / "metadata.json",
        DATA_DIR / "node_features.npy",
        DATA_DIR / "edge_features.npy",
        DATA_DIR / "edge_labels.npy",
        DATA_DIR / "valid_mask.npy",
    ]
    if not all(path.exists() for path in required):
        return None

    metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    return {
        "metadata": metadata,
        "node_features": np.load(DATA_DIR / "node_features.npy"),
        "edge_features": np.load(DATA_DIR / "edge_features.npy"),
        "edge_labels": np.load(DATA_DIR / "edge_labels.npy"),
        "valid_mask": np.load(DATA_DIR / "valid_mask.npy"),
    }


@st.cache_data(show_spinner=False)
def load_prediction_artifact() -> Optional[np.ndarray]:
    path = RESULTS_DIR / "test_predictions.npy"
    if not path.exists():
        return None
    return np.load(path)


@st.cache_data(show_spinner=False)
def load_results_metadata() -> Dict[str, Any]:
    path = RESULTS_DIR / "results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def heuristic_prediction_matrix(data: Dict[str, Any], period_index: int) -> np.ndarray:
    edge_features = data["edge_features"][period_index]
    conflict = edge_features[:, :, 0]
    cooperation = edge_features[:, :, 1]
    tone = edge_features[:, :, 2]

    conflict_norm = conflict / (conflict.max() + 1e-6)
    cooperation_relief = cooperation / (cooperation.max() + 1e-6)
    tone_penalty = 1 - ((np.tanh(tone / 8.0) + 1) / 2)

    risk = 0.62 * conflict_norm + 0.28 * tone_penalty - 0.18 * cooperation_relief
    risk = np.clip(risk, 0, 1)
    np.fill_diagonal(risk, 0.0)
    return risk


def resolve_prediction_matrix(data: Dict[str, Any], period_index: int) -> tuple[np.ndarray, str]:
    artifact = load_prediction_artifact()
    if artifact is None:
        return heuristic_prediction_matrix(data, period_index), "Heuristic estimate from latest features"

    results_metadata = load_results_metadata()
    artifact_periods = results_metadata.get("test_periods", [])
    if artifact.ndim == 2:
        matrix = artifact
    elif artifact.ndim == 3:
        selected_period = data["metadata"]["periods"][period_index]
        if selected_period in artifact_periods:
            mapped_index = artifact_periods.index(selected_period)
        else:
            mapped_index = artifact.shape[0] - 1
        matrix = artifact[mapped_index]
    else:
        return heuristic_prediction_matrix(data, period_index), "Fallback heuristic (unexpected prediction shape)"

    matrix = np.asarray(matrix, dtype=float)
    matrix = np.clip(matrix, 0, 1)
    np.fill_diagonal(matrix, 0.0)
    if artifact_periods:
        label = "Saved model predictions"
        if data["metadata"]["periods"][period_index] not in artifact_periods:
            label = f"Saved model predictions (latest available test period: {artifact_periods[-1]})"
        return matrix, label
    return matrix, "Saved model predictions"


@st.cache_resource(show_spinner=False)
def get_kg_builder() -> Optional[KnowledgeGraphBuilder]:
    if not HAS_OPTIONAL_BACKEND:
        return None
    return KnowledgeGraphBuilder(str(DATA_DIR))


@st.cache_resource(show_spinner=False)
def get_kg_exporter() -> Optional[KnowledgeGraphExporter]:
    if not HAS_OPTIONAL_BACKEND:
        return None
    return KnowledgeGraphExporter(str(VIS_DIR))


def format_period_label(period: str) -> str:
    year, month = period.split("-")
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    return f"{month_names[int(month) - 1]} {year}"


def sidebar(data: Dict[str, Any]) -> Dict[str, Any]:
    metadata = data["metadata"]
    periods = metadata["periods"]
    countries = list(metadata["country_indices"].keys())

    st.sidebar.title("Control Room")
    selected_period = st.sidebar.selectbox(
        "Analysis period",
        periods,
        index=len(periods) - 1,
        format_func=format_period_label,
    )
    focus_country = st.sidebar.selectbox("Country focus", ["All"] + countries, index=0)
    relation_mode = st.sidebar.radio(
        "Relationship lens",
        ["All ties", "Conflict only", "Cooperation only"],
        index=0,
    )

    llm_options = LLMFactory.available() if HAS_OPTIONAL_BACKEND else []
    provider = st.sidebar.selectbox("LLM provider", ["None"] + llm_options, index=0)
    st.session_state["llm_provider"] = None if provider == "None" else provider

    if HAS_OPTIONAL_BACKEND:
        st.sidebar.success("Knowledge graph and LLM hooks are available.")
    else:
        st.sidebar.warning("Optional backend features are unavailable in this environment.")

    return {
        "period": selected_period,
        "period_index": periods.index(selected_period),
        "countries": countries,
        "focus_country": focus_country,
        "relation_mode": relation_mode,
    }


def build_pair_table(
    countries: list[str],
    matrix: np.ndarray,
    edge_features: np.ndarray,
    focus_country: str,
    relation_mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, src in enumerate(countries):
        for j, dst in enumerate(countries):
            if i == j:
                continue
            if focus_country != "All" and focus_country not in {src, dst}:
                continue

            conflict = float(edge_features[i, j, 0])
            cooperation = float(edge_features[i, j, 1])
            if relation_mode == "Conflict only" and conflict <= 0:
                continue
            if relation_mode == "Cooperation only" and cooperation <= 0:
                continue

            rows.append(
                {
                    "Source": src,
                    "Target": dst,
                    "Risk": float(matrix[i, j]),
                    "Conflict events": conflict,
                    "Cooperation events": cooperation,
                    "Tone": float(edge_features[i, j, 2]),
                    "Goldstein": float(edge_features[i, j, 3]),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["Risk", "Conflict events"], ascending=[False, False]).reset_index(drop=True)


def render_metric_cards(data: Dict[str, Any], cfg: Dict[str, Any], prediction_matrix: np.ndarray, prediction_source: str) -> None:
    t = cfg["period_index"]
    edge_features = data["edge_features"][t]
    valid_mask = data["valid_mask"][t].astype(bool)
    conflict = edge_features[:, :, 0]
    cooperation = edge_features[:, :, 1]
    tone = edge_features[:, :, 2]

    top_idx = np.unravel_index(np.argmax(prediction_matrix), prediction_matrix.shape)
    countries = cfg["countries"]
    top_pair = f"{countries[top_idx[0]]} -> {countries[top_idx[1]]}"

    cards = st.columns(4)
    metrics = [
        ("Tracked countries", str(len(countries)), f"{len(data['metadata']['periods'])} monthly snapshots"),
        ("Total conflict edges", f"{int((conflict > 0).sum())}", f"Period: {format_period_label(cfg['period'])}"),
        ("Average tone", f"{tone[valid_mask].mean():.2f}", "Masked over valid country pairs"),
        ("Top risk pair", top_pair, prediction_source),
    ]

    for col, (label, value, caption) in zip(cards, metrics):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-caption">{caption}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def tab_overview(data: Dict[str, Any], cfg: Dict[str, Any], prediction_matrix: np.ndarray, prediction_source: str) -> None:
    t = cfg["period_index"]
    edge_features = data["edge_features"][t]
    node_features = data["node_features"][t]
    countries = cfg["countries"]

    render_metric_cards(data, cfg, prediction_matrix, prediction_source)
    st.write("")

    heatmap_col, table_col = st.columns([1.8, 1])
    with heatmap_col:
        fig = px.imshow(
            prediction_matrix,
            x=countries,
            y=countries,
            color_continuous_scale=["#f4f1de", "#d1495b", "#8b0000"],
            zmin=0,
            zmax=1,
            aspect="auto",
            title="Conflict risk matrix",
        )
        fig.update_layout(height=560, margin=dict(l=20, r=20, t=56, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        pair_table = build_pair_table(
            countries,
            prediction_matrix,
            edge_features,
            cfg["focus_country"],
            cfg["relation_mode"],
        )
        st.subheader("Highest-risk corridors")
        if pair_table.empty:
            st.info("No country pairs match the current filters.")
        else:
            show_df = pair_table.head(12).copy()
            show_df["Risk"] = show_df["Risk"].map(lambda value: f"{value:.3f}")
            show_df["Tone"] = show_df["Tone"].map(lambda value: f"{value:.2f}")
            show_df["Goldstein"] = show_df["Goldstein"].map(lambda value: f"{value:.2f}")
            st.dataframe(show_df, hide_index=True, use_container_width=True)

    activity_col, sentiment_col = st.columns(2)
    with activity_col:
        degree_df = pd.DataFrame(
            {
                "Country": countries,
                "Network degree": node_features[:, 5],
                "Region": [REGION_MAP.get(country, "Unknown") for country in countries],
            }
        ).sort_values("Network degree", ascending=False)
        fig = px.bar(
            degree_df,
            x="Country",
            y="Network degree",
            color="Region",
            color_discrete_map=REGION_COLORS,
            title="Country connectivity this period",
        )
        fig.update_layout(height=420, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with sentiment_col:
        valid_mask = data["valid_mask"][t].astype(bool)
        tone_values = edge_features[:, :, 2][valid_mask]
        fig = px.histogram(
            x=tone_values,
            nbins=24,
            title="Sentiment distribution across valid ties",
            labels={"x": "Average tone"},
            color_discrete_sequence=["#2d6cdf"],
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)


def tab_network(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    st.subheader("Knowledge graph snapshot")
    if not HAS_OPTIONAL_BACKEND:
        st.info("Optional knowledge graph modules are not available in the active Python environment.")
        if BACKEND_IMPORT_ERROR:
            st.caption(BACKEND_IMPORT_ERROR)
        return

    builder = get_kg_builder()
    exporter = get_kg_exporter()
    if builder is None or exporter is None:
        st.info("Knowledge graph backend could not be initialized.")
        return

    graph = builder.build_graph_for_period(cfg["period"])
    summary = builder.get_summary(cfg["period"])
    html = exporter.to_pyvis_html(graph, cfg["period"], title=f"KG {cfg['period']}")

    graph_col, info_col = st.columns([1.75, 1])
    with graph_col:
        st.components.v1.html(html, height=720, scrolling=True)

    with info_col:
        st.metric("Graph nodes", summary["num_nodes"])
        st.metric("Graph edges", summary["num_edges"])
        st.metric("Density", f"{summary['density']:.3f}")
        st.json(summary, expanded=False)

        top_edges: list[tuple[str, str, float, str]] = []
        for source, target, payload in graph.edges(data=True):
            edge_type = payload.get("edge_type")
            if edge_type in {"CONFLICT_WITH", "COOPERATE_WITH"}:
                top_edges.append((source, target, float(payload.get("weight", 0.0)), edge_type))

        top_edges.sort(key=lambda item: item[2], reverse=True)
        st.markdown("#### Strongest visible ties")
        for source, target, weight, edge_type in top_edges[:10]:
            label = "Conflict" if edge_type == "CONFLICT_WITH" else "Cooperation"
            st.write(f"{label}: {source} -> {target} ({weight:.0f})")


def tab_timeline(data: Dict[str, Any], cfg: Dict[str, Any], prediction_matrix: np.ndarray) -> None:
    st.subheader("Pair timeline")
    countries = cfg["countries"]

    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Source country", countries, index=0)
    with col2:
        target_choices = [country for country in countries if country != source]
        target = st.selectbox("Target country", target_choices, index=0)

    source_idx = data["metadata"]["country_indices"][source]
    target_idx = data["metadata"]["country_indices"][target]
    periods = [format_period_label(period) for period in data["metadata"]["periods"]]
    edge_features = data["edge_features"]

    conflict_series = edge_features[:, source_idx, target_idx, 0]
    cooperation_series = edge_features[:, source_idx, target_idx, 1]
    tone_series = edge_features[:, source_idx, target_idx, 2]
    goldstein_series = edge_features[:, source_idx, target_idx, 3]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Interaction volumes", "Tone and Goldstein trend"),
        vertical_spacing=0.16,
    )
    fig.add_trace(go.Scatter(x=periods, y=conflict_series, mode="lines+markers", name="Conflict", line=dict(color="#d1495b", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=periods, y=cooperation_series, mode="lines+markers", name="Cooperation", line=dict(color="#2a9d8f", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=periods, y=tone_series, mode="lines+markers", name="Tone", line=dict(color="#2d6cdf", width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=periods, y=goldstein_series, mode="lines+markers", name="Goldstein", line=dict(color="#c89100", width=3)), row=2, col=1)
    fig.update_layout(height=640, margin=dict(l=20, r=20, t=52, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Latest displayed risk for {source} -> {target}: {prediction_matrix[source_idx, target_idx]:.3f}"
    )


def tab_global_view(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    st.subheader("Global network posture")
    t = cfg["period_index"]
    countries = cfg["countries"]
    edge_features = data["edge_features"][t]
    node_features = data["node_features"][t]

    fig = go.Figure()
    latitudes: list[float] = []
    longitudes: list[float] = []
    colors: list[str] = []
    sizes: list[float] = []
    labels: list[str] = []

    for country in countries:
        latitude, longitude = COUNTRY_COORDS.get(country, (0.0, 0.0))
        idx = data["metadata"]["country_indices"][country]
        latitudes.append(latitude)
        longitudes.append(longitude)
        sizes.append(8 + min(26, float(node_features[idx, 5]) * 0.8))
        region = REGION_MAP.get(country, "Unknown")
        colors.append(REGION_COLORS.get(region, "#6b7280"))
        labels.append(f"{country}<br>Region: {region}<br>Degree: {node_features[idx, 5]:.0f}")

    fig.add_trace(
        go.Scattergeo(
            lat=latitudes,
            lon=longitudes,
            text=labels,
            mode="markers+text",
            textposition="top center",
            marker=dict(size=sizes, color=colors, opacity=0.88, line=dict(width=0.8, color="#ffffff")),
            hoverinfo="text",
            name="Countries",
        )
    )

    for i, source in enumerate(countries):
        for j, target in enumerate(countries):
            if i == j:
                continue
            conflict_weight = float(edge_features[i, j, 0])
            if conflict_weight <= 1.0:
                continue
            lat0, lon0 = COUNTRY_COORDS.get(source, (0.0, 0.0))
            lat1, lon1 = COUNTRY_COORDS.get(target, (0.0, 0.0))
            fig.add_trace(
                go.Scattergeo(
                    lat=[lat0, lat1],
                    lon=[lon0, lon1],
                    mode="lines",
                    line=dict(width=min(6, 1.2 + conflict_weight * 0.35), color="#d1495b"),
                    opacity=0.45,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=10, b=0),
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="#ebe4d7",
            showocean=True,
            oceancolor="#d9ebff",
            showcountries=True,
            countrycolor="#7f8c8d",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def tab_ai(data: Dict[str, Any], cfg: Dict[str, Any], prediction_matrix: np.ndarray) -> None:
    st.subheader("AI analyst")
    provider = st.session_state.get("llm_provider")
    if not provider:
        st.info("Select an LLM provider in the sidebar to generate explanations.")
        return

    countries = cfg["countries"]
    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Country A", countries, index=0)
    with col2:
        target_options = [country for country in countries if country != source]
        target = st.selectbox("Country B", target_options, index=0)

    i = data["metadata"]["country_indices"][source]
    j = data["metadata"]["country_indices"][target]
    edge_features = data["edge_features"][cfg["period_index"]]
    pair_data = {
        "source": source,
        "target": target,
        "risk": float(prediction_matrix[i, j]),
        "conflict_count": int(edge_features[i, j, 0]),
        "coop_count": int(edge_features[i, j, 1]),
        "avg_tone": float(edge_features[i, j, 2]),
        "avg_goldstein": float(edge_features[i, j, 3]),
        "period": cfg["period"],
    }

    st.json(pair_data, expanded=False)
    if st.button("Generate risk brief", use_container_width=True):
        with st.spinner("Building analyst note..."):
            try:
                llm = LLMFactory.create(provider)
                report = llm.generate_risk_report(pair_data)
                st.markdown(report)
            except Exception as exc:
                st.error(f"LLM error: {exc}")

    question = st.text_area(
        "Ask a follow-up question",
        placeholder="Why does this pair look risky right now?",
        height=90,
    )
    if st.button("Ask model", use_container_width=True) and question:
        with st.spinner("Generating answer..."):
            try:
                llm = LLMFactory.create(provider)
                prompt = (
                    f"Period: {cfg['period']}. Pair: {source} -> {target}. "
                    f"Risk={pair_data['risk']:.3f}, conflict={pair_data['conflict_count']}, "
                    f"cooperation={pair_data['coop_count']}, tone={pair_data['avg_tone']:.2f}, "
                    f"goldstein={pair_data['avg_goldstein']:.2f}. "
                    f"Question: {question}"
                )
                answer = llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
                st.markdown(answer)
            except Exception as exc:
                st.error(f"LLM error: {exc}")


def tab_live_feed() -> None:
    st.subheader("Live GDELT feed")
    try:
        from src.gdelt import GDELTFetcher
    except Exception as exc:  # pragma: no cover - network feature
        st.info("The live feed module is unavailable in the active environment.")
        st.caption(str(exc))
        return

    query = st.text_input("Search query", value="geopolitical conflict")
    max_rows = st.slider("Max articles", min_value=5, max_value=30, value=10)

    if st.button("Fetch articles", use_container_width=True):
        with st.spinner("Querying GDELT..."):
            try:
                articles = GDELTFetcher.fetch_by_queries([query], max_rows=max_rows)
            except Exception as exc:
                st.error(f"Fetch error: {exc}")
                return

        if not articles:
            st.warning("No articles returned for that query.")
            return

        for article in articles:
            st.markdown(f"#### [{article['title']}]({article['url']})")
            st.caption(
                f"{article.get('source', 'Unknown')} | "
                f"{article.get('published_at', 'Unknown time')} | "
                f"Tone: {article.get('tone', 'N/A')}"
            )
            st.markdown("---")


def main() -> None:
    inject_styles()
    data = load_processed_data()

    st.markdown(
        """
        <div class="hero">
            <h1>Geopolitical Conflict Pulse</h1>
            <p>
                Explore temporal country networks, inspect conflict pressure, and review model-driven
                risk signals from processed GDELT event data in one browser dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data is None:
        st.error("Processed demo data was not found. Run `python scripts/setup_demo.py` first.")
        return

    cfg = sidebar(data)
    prediction_matrix, prediction_source = resolve_prediction_matrix(data, cfg["period_index"])

    tabs = st.tabs(
        [
            "Overview",
            "Knowledge Graph",
            "Timeline",
            "Global View",
            "AI Analyst",
            "Live Feed",
        ]
    )

    with tabs[0]:
        tab_overview(data, cfg, prediction_matrix, prediction_source)
    with tabs[1]:
        tab_network(data, cfg)
    with tabs[2]:
        tab_timeline(data, cfg, prediction_matrix)
    with tabs[3]:
        tab_global_view(data, cfg)
    with tabs[4]:
        tab_ai(data, cfg, prediction_matrix)
    with tabs[5]:
        tab_live_feed()


if __name__ == "__main__":
    main()
