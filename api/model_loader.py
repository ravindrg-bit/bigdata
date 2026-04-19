from __future__ import annotations

import json
from pathlib import Path

import joblib
import shap


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hybrid_ensemble.pkl"
THRESHOLD_PATH = ROOT / "models" / "hybrid_ensemble.threshold.json"


FEATURE_LABELS = {
    "borrower_id": "Borrower ID",
    "group_id": "Group ID",
    "prior_CMR_usage": "Prior CMR usage",
    "store_visit_count": "Falabella store visits",
    "CoDi_wallet_flag": "CoDi wallet active",
    "INE_verified_flag": "INE identity verified",
    "rural_flag": "Rural residence",
    "indigenous_proxy": "Indigenous proxy",
    "age": "Borrower age",
    "cycle_number": "Group lending cycle",
    "cohesion_score": "Group cohesion",
    "loan_count": "Loan count",
    "avg_loan_amount_MXN": "Average loan amount",
    "max_loan_amount_MXN": "Maximum loan amount",
    "cmr_credit_line_share": "CMR credit line share",
    "oxxo_cash_backed_share": "OXXO cash-backed share",
    "repayment_latency_days": "Repayment latency",
    "on_time_repayment_share": "On-time repayment history",
    "routine_score": "Call routine consistency",
    "messaging_frequency": "Messaging frequency",
    "call_volume_stability": "Call volume stability",
    "app_opens": "App opens",
    "location_variance": "Location variance",
    "Falabella_app_session_flag": "Falabella app active",
    "routine_entropy": "Behavioral entropy",
    "CoDi_transaction_regularity": "CoDi transaction regularity",
    "falabella_app_session_recency": "Falabella app session recency",
    "degree_centrality": "Network connections",
    "weighted_tie_strength": "Peer tie strength",
    "betweenness_centrality": "Network broker position",
    "neighborhood_default_rate_1hop": "Neighbour default rate",
    "neighborhood_default_rate_2hop": "Second-hop neighbour default rate",
    "pagerank_score": "Social network influence",
    "community_membership_flag": "Community member flag",
    "default_flag": "Observed default flag",
    "sequential_lending_flag": "Sequential lending flag",
    "peer_default_contagion_score": "Peer default contagion",
    "gender_female_flag": "Female borrower flag",
}


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Hybrid model not found: {MODEL_PATH}")

_MODEL = joblib.load(MODEL_PATH)
_EXPLAINER = shap.TreeExplainer(_MODEL)
_DECISION_THRESHOLD = 0.5

if THRESHOLD_PATH.exists():
    payload = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    _DECISION_THRESHOLD = float(payload.get("decision_threshold", 0.5))


def get_model():
    return _MODEL


def get_explainer():
    return _EXPLAINER


def get_decision_threshold() -> float:
    return _DECISION_THRESHOLD


def get_feature_label(feature_name: str) -> str:
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    if feature_name.startswith("emb_"):
        return f"Graph embedding {feature_name.split('_', 1)[1]}"
    return feature_name.replace("_", " ").title()
