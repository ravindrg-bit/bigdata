from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from falabella_risk.app.dashboard_components import (
    build_network_risk_graph,
    build_phase_transition_chart,
    build_risk_histogram,
    build_shap_waterfall,
)
from falabella_risk.evaluation.explainability import ExplainabilityEngine
from falabella_risk.federated.federated_training import FederatedInferenceEngine
from falabella_risk.inference.cold_start import ColdStartScorer

st.set_page_config(page_title="Falabella Risk Engine", layout="wide")

st.title("Falabella Hybrid Risk Engine")
st.caption("MVP implementation: portfolio analytics and borrower lookup are active.")

features_path = Path("data/processed/features.parquet")
month2_model_path = Path("models/baseline_xgb.pkl")
month3_model_path = Path("models/hybrid_ensemble.pkl")
embeddings_path = Path("data/processed/gnn_embeddings.parquet")
edges_path = Path("data/raw/edges.parquet")
federated_model_path = Path("models/federated_model.pt")
federated_report_path = Path("reports/federated_report.json")
fairness_report_path = Path("reports/fairness_report.json")
override_log_path = Path("reports/override_log.csv")
model_card_path = Path("docs/model_card.md")

if not features_path.exists():
    st.warning(
        "data/processed/features.parquet not found. Run data generation and feature engineering first."
    )
    st.code("python -m falabella_risk.data.generate_data --seed 42 --output-dir data/raw")
    st.code(
        "python -m falabella_risk.features.feature_engineering --data-dir data/raw --output data/processed/features.parquet"
    )
    st.stop()

features = pd.read_parquet(features_path)

with st.sidebar:
    st.header("Controls")
    model_toggle = st.selectbox("Model mode", ["Grameen-only", "Tala-only", "Hybrid"], index=2)
    sample_size = st.slider(
        "Portfolio sample size", min_value=100, max_value=5000, value=1200, step=100
    )
    phase_view = st.selectbox(
        "Rollout phase view",
        ["Month 1 Rule-Based", "Months 2-3 Hybrid-Lite", "Month 3+ Full Hybrid"],
        index=2,
    )
    federated_toggle = st.toggle("Federated mode", value=False)
    st.plotly_chart(build_phase_transition_chart(), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Borrowers", f"{len(features):,}")
col2.metric("Default Rate", f"{features['default_flag'].mean() * 100:.2f}%")
col3.metric("Features", f"{features.shape[1] - 2}")

try:
    scorer = ColdStartScorer(
        month2_model_path=month2_model_path if month2_model_path.exists() else None,
        month3_model_path=month3_model_path if month3_model_path.exists() else None,
    )
except Exception:
    scorer = ColdStartScorer(month2_model_path=None, month3_model_path=None)

explain_engine = None
if month3_model_path.exists():
    try:
        explain_engine = ExplainabilityEngine(
            model_path=month3_model_path,
            embeddings_path=embeddings_path if embeddings_path.exists() else None,
            threshold=0.5,
        )
    except Exception:
        explain_engine = None

federated_engine = None
if federated_toggle and federated_model_path.exists():
    try:
        federated_engine = FederatedInferenceEngine(model_path=federated_model_path)
    except Exception:
        federated_engine = None

tab1, tab2, tab3 = st.tabs(["Portfolio Risk", "Borrower Lookup", "Ethics Audit"])

with tab1:
    st.subheader("Portfolio Risk Distribution")
    portfolio_sample = features.sample(min(sample_size, len(features)), random_state=42)

    hist_fig = build_risk_histogram(portfolio_sample)
    if hist_fig is not None:
        st.plotly_chart(hist_fig, use_container_width=True)

    network_fig = build_network_risk_graph(
        features=features,
        edges_path=edges_path,
        sample_size=min(500, sample_size),
        seed=42,
    )
    if network_fig is not None:
        st.plotly_chart(network_fig, use_container_width=True)

    if {
        "neighborhood_default_rate_1hop",
        "repayment_latency_days",
        "default_flag",
    }.issubset(portfolio_sample.columns):
        scatter = portfolio_sample.copy()
        scatter["default_label"] = scatter["default_flag"].map({0.0: "non_default", 1.0: "default"})
        scatter_fig = px.scatter(
            scatter,
            x="neighborhood_default_rate_1hop",
            y="repayment_latency_days",
            color="default_label",
            title="Risk Signal Check: Neighborhood Default vs Repayment Latency",
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

with tab2:
    st.subheader("Single Borrower Lookup")
    input_mode = st.radio(
        "Input Source", ["Existing Borrower", "Manual Applicant Form"], horizontal=True
    )

    if input_mode == "Existing Borrower":
        borrower_id = st.selectbox(
            "Borrower ID", options=features["borrower_id"].head(5000).tolist()
        )
        row = features.loc[features["borrower_id"] == borrower_id].iloc[0].copy()
    else:
        base_row = features.median(numeric_only=True)
        row = pd.Series(base_row)
        row["borrower_id"] = -1
        row["default_flag"] = 0

        curp_hash = st.text_input("CURP hash", value="MANUAL_CURP_HASH")
        ine_verified = st.selectbox("INE verified", [0, 1], index=1)
        store_visits = st.number_input(
            "Store visit count", min_value=0, max_value=50, value=4, step=1
        )
        codi_wallet = st.selectbox("CoDi wallet flag", [0, 1], index=1)
        rural_flag = st.selectbox("Rural flag", [0, 1], index=0)
        indigenous_proxy = st.selectbox("Indigenous proxy", [0, 1], index=0)
        call_volume = st.number_input("Call volume", min_value=0, max_value=1000, value=80, step=1)
        routine_score = st.slider(
            "Routine score", min_value=0.0, max_value=1.0, value=0.6, step=0.01
        )
        messaging_frequency = st.number_input(
            "Messaging frequency", min_value=0, max_value=1000, value=42, step=1
        )

        row["CURP_hash"] = curp_hash
        row["INE_verified_flag"] = float(ine_verified)
        row["store_visit_count"] = float(store_visits)
        row["CoDi_wallet_flag"] = float(codi_wallet)
        row["rural_flag"] = float(rural_flag)
        row["indigenous_proxy"] = float(indigenous_proxy)
        row["call_volume"] = float(call_volume)
        row["routine_score"] = float(routine_score)
        row["messaging_frequency"] = float(messaging_frequency)

    score_payload = scorer.score(row)

    if federated_engine is not None:
        fed_risk = federated_engine.score_row(row)
        score_payload["risk_score"] = fed_risk
        score_payload["decision"] = "approve" if fed_risk < 0.5 else "decline"
        score_payload["model_source"] = "federated_fedavg"
        low, high = scorer._credit_line(score_payload["phase"], fed_risk)
        score_payload["credit_line_range_mxn"] = [low, high]

    risk_score = score_payload["risk_score"]
    decision = score_payload["decision"]
    credit_low, credit_high = score_payload["credit_line_range_mxn"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", f"{risk_score * 100:.2f}")
    c2.metric("Decision", decision.upper())
    c3.metric("Assigned Phase", score_payload["phase"])
    c4.metric("Credit Line", f"MXN {credit_low:,} - {credit_high:,}")
    st.progress(float(min(max(risk_score, 0.0), 1.0)), text=f"Risk Gauge: {risk_score * 100:.2f}%")

    st.caption(
        f"Model source: {score_payload['model_source']} | Active sidebar mode: {model_toggle} | Phase view: {phase_view}"
    )

    if explain_engine is not None:
        explanation = explain_engine.predict_explain(
            row.drop(labels=["default_flag"], errors="ignore")
        )
        st.write("Top explainability drivers")
        st.dataframe(pd.DataFrame(explanation["top_drivers"]), use_container_width=True)
        st.caption(f"Explainability latency: {explanation['latency_ms']:.2f} ms")
        waterfall = build_shap_waterfall(
            top_drivers=explanation["top_drivers"],
            threshold=float(explanation["threshold"]),
            risk_score=float(explanation["risk_score"]),
        )
        st.plotly_chart(waterfall, use_container_width=True)

    st.write("Borrower feature snapshot")
    st.dataframe(row.to_frame().T, use_container_width=True)

    st.write("Loan Officer Override")
    override_decision = st.selectbox("Override decision", ["no_override", "approve", "decline"])
    override_reason = st.text_area("Override reason", value="")

    if st.button("Log Override Decision"):
        override_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = pd.DataFrame(
            [
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "borrower_id": int(row.get("borrower_id", -1)),
                    "model_source": score_payload["model_source"],
                    "predicted_decision": decision,
                    "override_decision": override_decision,
                    "reason": override_reason,
                    "risk_score": float(risk_score),
                }
            ]
        )

        if override_log_path.exists():
            existing = pd.read_csv(override_log_path)
            out = pd.concat([existing, record], ignore_index=True)
        else:
            out = record
        out.to_csv(override_log_path, index=False)
        st.success("Override saved.")

    if override_log_path.exists():
        st.caption("Recent override log entries")
        st.dataframe(pd.read_csv(override_log_path).tail(15), use_container_width=True)

with tab3:
    st.subheader("Ethics Audit")

    if fairness_report_path.exists():
        report = json.loads(fairness_report_path.read_text(encoding="utf-8"))
        mitigation = report.get("mitigation", {}).get("fairness", {})

        rows = []
        for attr, rec in mitigation.items():
            rows.append(
                {
                    "attribute": attr,
                    "demographic_parity_gap": rec.get("demographic_parity_gap"),
                    "equal_opportunity_gap": rec.get("equal_opportunity_gap"),
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric(
            "Parity Threshold",
            f"{report.get('pass_criteria', {}).get('parity_gap_threshold', 0.05) * 100:.1f}%",
        )
        c2.metric(
            "All Parity Gaps < Threshold",
            str(report.get("pass_criteria", {}).get("all_parity_gaps_below_threshold", False)),
        )
    else:
        st.warning(
            "Fairness report not found. Run python -m falabella_risk.evaluation.fairness_audit first."
        )

    if explain_engine is not None:
        sample_rows = features.sample(min(300, len(features)), random_state=42)
        try:
            importance = explain_engine.global_feature_importance(sample_rows, max_rows=300).head(
                20
            )
            if not importance.empty:
                fig = px.bar(
                    importance.sort_values("mean_abs_shap", ascending=True),
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    title="Global Feature Importance (Mean |SHAP|)",
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Feature importance chart unavailable for current model state.")

    if model_card_path.exists():
        st.download_button(
            label="Download Model Card",
            data=model_card_path.read_text(encoding="utf-8"),
            file_name="model_card.md",
            mime="text/markdown",
        )

    if federated_report_path.exists():
        fed = json.loads(federated_report_path.read_text(encoding="utf-8"))
        st.caption("Federated vs centralized benchmark")
        st.json(
            {
                "federated_auc": fed.get("federated_metrics", {}).get("auc"),
                "centralized_auc": fed.get("centralized_metrics", {}).get("auc"),
                "auc_gap_vs_centralized": fed.get("auc_gap_vs_centralized"),
            }
        )
