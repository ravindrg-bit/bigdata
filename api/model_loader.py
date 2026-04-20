from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import shap


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hybrid_ensemble.pkl"
SCORE_PERCENTILES_PATH = ROOT / "models" / "score_percentiles.npy"

CAUTION_THRESHOLD_PERCENTILE = 0.60
DECLINE_THRESHOLD_PERCENTILE = 0.80
REFERENCE_AUC = 0.98


FEATURE_LABELS = {
    "borrower_id": "Borrower ID",
    "group_id": "Lending group ID",
    "prior_CMR_usage": "Prior store credit usage",
    "store_visit_count": "Falabella store visits",
    "CoDi_wallet_flag": "CoDi wallet active",
    "INE_verified_flag": "INE identity verified",
    "rural_flag": "Rural location",
    "indigenous_proxy": "Indigenous community",
    "age": "Age",
    "cycle_number": "Lending group cycle",
    "cohesion_score": "Group cohesion",
    "loan_count": "Number of loans",
    "avg_loan_amount_MXN": "Average loan amount",
    "max_loan_amount_MXN": "Maximum loan amount",
    "avg_loan_amount": "Average loan amount",
    "max_loan_amount": "Maximum loan amount",
    "total_loan_amount": "Total loan amount",
    "CMR_credit_line_share": "CMR credit line share",
    "OXXO_cash_backed_share": "OXXO cash-backed share",
    "cmr_credit_line_share": "CMR credit line share",
    "oxxo_cash_backed_share": "OXXO cash-backed share",
    "repayment_latency_days": "Average repayment delay (days)",
    "mean_repayment_latency": "Average repayment delay (days)",
    "on_time_repayment_share": "On-time repayment rate",
    "worst_latency": "Worst repayment delay (days)",
    "total_repayments": "Number of repayments",
    "call_volume": "Monthly call volume",
    "routine_score": "Call routine consistency",
    "call_routine_score": "Call routine consistency",
    "messaging_frequency": "Messaging frequency",
    "call_volume_stability": "Call volume stability",
    "weekly_call_cv": "Call volume stability",
    "app_opens": "App opens per month",
    "location_variance": "Location variance",
    "Falabella_app_session_flag": "Falabella app user",
    "routine_entropy": "Daily routine predictability",
    "CoDi_transaction_regularity": "CoDi transaction regularity",
    "codi_txn_regularity": "CoDi transaction regularity",
    "falabella_app_session_recency": "Days since last app session",
    "app_session_recency_days": "Days since last app session",
    "degree_centrality": "Network connections (centrality)",
    "weighted_tie_strength": "Social tie strength",
    "betweenness_centrality": "Network broker position",
    "neighborhood_default_rate_1hop": "Neighbour default rate",
    "neighborhood_default_rate_2hop": "Extended network default rate",
    "pagerank_score": "Social network influence",
    "community_membership_flag": "Community cluster ID",
    "community_id": "Community cluster ID",
    "default_flag": "Observed default flag",
    "sequential_lending_flag": "Sequential lending active",
    "peer_default_contagion_score": "Peer default contagion",
    "gender_female_flag": "Gender (female)",
}

GRAPH_SIGNAL_LABEL = "Social network position (graph analysis)"


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Hybrid model not found: {MODEL_PATH}")

if not SCORE_PERCENTILES_PATH.exists():
    raise FileNotFoundError(f"Score percentiles not found: {SCORE_PERCENTILES_PATH}")

_MODEL = joblib.load(MODEL_PATH)
_EXPLAINER = shap.TreeExplainer(_MODEL)
_SCORE_PERCENTILES = np.load(SCORE_PERCENTILES_PATH)

if _SCORE_PERCENTILES.ndim != 1 or _SCORE_PERCENTILES.size == 0:
    raise ValueError("score_percentiles.npy must be a non-empty 1-D array")


def get_model():
    return _MODEL


def get_explainer():
    return _EXPLAINER


def score_to_percentile(raw_score: float) -> float:
    idx = int(np.searchsorted(_SCORE_PERCENTILES, float(raw_score), side="left"))
    idx = max(0, min(idx, int(_SCORE_PERCENTILES.size)))
    return round(idx / float(_SCORE_PERCENTILES.size), 4)


def percentile_to_decision(percentile: float) -> tuple[str, str]:
    pct = float(percentile)
    if pct < CAUTION_THRESHOLD_PERCENTILE:
        return "approve", "low"
    if pct < DECLINE_THRESHOLD_PERCENTILE:
        return "approve_with_conditions", "medium"
    return "decline", "high"


def get_model_info() -> dict[str, float | int | str]:
    return {
        "scoring_method": "percentile_rank",
        "decline_threshold_percentile": DECLINE_THRESHOLD_PERCENTILE,
        "caution_threshold_percentile": CAUTION_THRESHOLD_PERCENTILE,
        "total_population": int(_SCORE_PERCENTILES.size),
        "auc": REFERENCE_AUC,
    }


def get_feature_label(feature_name: str) -> str:
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    if feature_name.startswith("emb_"):
        return f"Graph embedding {feature_name.split('_', 1)[1]}"
    return feature_name.replace("_", " ").title()


def get_graph_signal_label() -> str:
    return GRAPH_SIGNAL_LABEL
