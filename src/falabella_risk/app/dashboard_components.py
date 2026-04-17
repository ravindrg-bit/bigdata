from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_risk_histogram(features: pd.DataFrame):
    if "default_flag" not in features.columns:
        return None

    df = features.copy()
    df["default_label"] = df["default_flag"].map({0.0: "non_default", 1.0: "default"})

    return px.histogram(
        df,
        x="neighborhood_default_rate_1hop",
        color="default_label",
        barmode="overlay",
        nbins=40,
        title="Neighborhood Default Rate Distribution",
        labels={
            "neighborhood_default_rate_1hop": "Neighborhood Default Rate (1-hop)",
            "default_label": "Default Label",
        },
        opacity=0.75,
    )


def build_phase_transition_chart() -> go.Figure:
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=18,
                    line=dict(color="black", width=0.4),
                    label=[
                        "Store Visits >= 3",
                        "CoDi + Mobile Active",
                        "Peer Graph Available",
                        "Month 1 Rule-Based",
                        "Months 2-3 Hybrid-Lite",
                        "Month 3+ Full Hybrid",
                    ],
                ),
                link=dict(
                    source=[0, 1, 2],
                    target=[3, 4, 5],
                    value=[1, 1, 1],
                ),
            )
        ]
    )
    fig.update_layout(title_text="Phase Transition Logic", font_size=12, height=320)
    return fig


def build_network_risk_graph(
    features: pd.DataFrame,
    edges_path: Path,
    sample_size: int = 500,
    seed: int = 42,
) -> go.Figure | None:
    if not edges_path.exists():
        return None

    sample = features.sample(min(sample_size, len(features)), random_state=seed).copy()
    nodes = set(sample["borrower_id"].astype(int).tolist())

    edges = pd.read_parquet(edges_path)
    edge_view = edges[
        edges["src_id"].astype(int).isin(nodes) & edges["dst_id"].astype(int).isin(nodes)
    ].copy()

    if edge_view.empty:
        return None

    if len(edge_view) > 2200:
        edge_view = edge_view.sample(2200, random_state=seed)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(
        edge_view[["src_id", "dst_id", "tie_strength"]].itertuples(index=False, name=None)
    )

    layout = nx.spring_layout(graph, seed=seed, k=0.2)

    edge_x: list[float] = []
    edge_y: list[float] = []
    for src, dst in graph.edges():
        x0, y0 = layout[src]
        x1, y1 = layout[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.35, color="#A0A0A0"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    degree_map = dict(graph.degree())
    sample = sample.set_index("borrower_id")

    node_x: list[float] = []
    node_y: list[float] = []
    node_color: list[float] = []
    node_size: list[float] = []
    hover_text: list[str] = []

    for node in graph.nodes():
        x, y = layout[node]
        node_x.append(float(x))
        node_y.append(float(y))

        risk = (
            float(sample.loc[node, "neighborhood_default_rate_1hop"])
            if node in sample.index
            else 0.0
        )
        amount = (
            float(sample.loc[node, "avg_loan_amount_MXN"])
            if "avg_loan_amount_MXN" in sample.columns and node in sample.index
            else 2000.0
        )

        node_color.append(risk)
        node_size.append(float(np.clip(6 + (amount / 1500.0), 6, 22)))
        hover_text.append(
            f"borrower_id={node}<br>risk_proxy={risk:.3f}<br>degree={degree_map.get(node, 0)}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=hover_text,
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Risk Proxy"),
            line=dict(width=0.4, color="#222"),
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Portfolio Peer Network (Sampled)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=640,
    )
    return fig


def build_shap_waterfall(
    top_drivers: list[dict[str, object]], threshold: float, risk_score: float
) -> go.Figure:
    labels = [str(item["label"]) for item in top_drivers]
    effects = [float(item["shap_contribution"]) for item in top_drivers]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(effects),
            x=["Decision Threshold"] + labels,
            y=[float(threshold)] + effects,
            connector={"line": {"color": "rgb(120,120,120)"}},
        )
    )
    fig.add_hline(y=risk_score, line_dash="dot", line_color="#CC0000")
    fig.update_layout(
        title="SHAP Waterfall (Top Drivers)",
        yaxis_title="Risk Contribution",
        height=420,
        margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig
