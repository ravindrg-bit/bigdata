from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

try:
    from .schemas import BorrowerInput
except ImportError:
    from schemas import BorrowerInput


# Exact features.parquet column order (38 columns).
FEATURE_COLUMNS = [
    "borrower_id",
    "group_id",
    "prior_CMR_usage",
    "store_visit_count",
    "CoDi_wallet_flag",
    "INE_verified_flag",
    "rural_flag",
    "indigenous_proxy",
    "age",
    "cycle_number",
    "cohesion_score",
    "loan_count",
    "avg_loan_amount_MXN",
    "max_loan_amount_MXN",
    "cmr_credit_line_share",
    "oxxo_cash_backed_share",
    "repayment_latency_days",
    "on_time_repayment_share",
    "routine_score",
    "messaging_frequency",
    "call_volume_stability",
    "app_opens",
    "location_variance",
    "Falabella_app_session_flag",
    "routine_entropy",
    "CoDi_transaction_regularity",
    "falabella_app_session_recency",
    "degree_centrality",
    "weighted_tie_strength",
    "betweenness_centrality",
    "neighborhood_default_rate_1hop",
    "neighborhood_default_rate_2hop",
    "pagerank_score",
    "community_membership_flag",
    "default_flag",
    "sequential_lending_flag",
    "peer_default_contagion_score",
    "gender_female_flag",
]

INT_FEATURES = {
    "borrower_id",
    "group_id",
    "prior_CMR_usage",
    "store_visit_count",
    "CoDi_wallet_flag",
    "INE_verified_flag",
    "rural_flag",
    "indigenous_proxy",
    "age",
    "cycle_number",
    "loan_count",
    "app_opens",
    "Falabella_app_session_flag",
    "falabella_app_session_recency",
    "community_membership_flag",
    "default_flag",
    "sequential_lending_flag",
    "gender_female_flag",
}

TRACE_SOURCE_TYPES = {
    "direct_from_form",
    "derived_from_form",
    "training_set_median",
    "existing_profile",
    "existing_precomputed",
}


def _cast_feature_value(feature_name: str, value: float | int) -> float | int:
    if feature_name in INT_FEATURES:
        return int(round(float(value)))
    return float(value)


def _build_trace_row(
    index: int,
    feature_name: str,
    value: float | int,
    source_type: str,
    derived_from_fields: list[str] | None = None,
    derivation_formula: str | None = None,
    default_reason: str | None = None,
    median_value_used: float | None = None,
) -> dict[str, Any]:
    if source_type not in TRACE_SOURCE_TYPES:
        raise ValueError(f"Invalid source_type for {feature_name}: {source_type}")

    return {
        "index": int(index),
        "feature_name": feature_name,
        "value": _cast_feature_value(feature_name, value),
        "source_type": source_type,
        "derived_from_fields": list(derived_from_fields or []),
        "derivation_formula": derivation_formula,
        "default_reason": default_reason,
        "median_value_used": median_value_used,
    }


def _assert_trace_integrity(
    feature_values: OrderedDict[str, float | int],
    trace_rows: list[dict[str, Any]],
) -> None:
    if list(feature_values.keys()) != FEATURE_COLUMNS:
        raise ValueError("Feature construction must follow FEATURE_COLUMNS canonical order")

    if len(feature_values) != 38 or len(trace_rows) != 38:
        raise ValueError(
            f"Expected exactly 38 tabular features and 38 trace rows; got "
            f"{len(feature_values)} features and {len(trace_rows)} trace rows"
        )

    for row in trace_rows:
        if not row.get("source_type"):
            raise ValueError(f"Missing source_type in trace row: {row}")


def _median_or_fallback(
    feature_name: str,
    training_medians: dict[str, float],
    fallback: float = 0.0,
) -> float:
    if feature_name in training_medians:
        return float(training_medians[feature_name])
    return float(fallback)


def compute_training_feature_medians(features_table: pd.DataFrame) -> dict[str, float]:
    medians: dict[str, float] = {}
    for feature_name in FEATURE_COLUMNS:
        if feature_name == "borrower_id":
            continue
        if feature_name not in features_table.columns:
            continue
        numeric = pd.to_numeric(features_table[feature_name], errors="coerce")
        value = float(numeric.median()) if numeric.notna().any() else 0.0
        medians[feature_name] = value
    return medians


def build_manual_feature_bundle(
    input_data: BorrowerInput,
    training_medians: dict[str, float],
    graph_verification: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], pd.DataFrame]:
    """Build canonical manual tabular vector plus full per-feature trace metadata.

    The returned vector includes all 38 tabular features in `FEATURE_COLUMNS` order.
    """

    d = input_data.model_dump()
    graph_ctx = graph_verification or {}
    graph_verified = bool(graph_ctx.get("verified", False))
    validated_peer_count = int(graph_ctx.get("validated_peer_count", 0) or 0)
    validated_rate_1hop = graph_ctx.get("validated_neighborhood_default_rate_1hop")

    if graph_verified and validated_rate_1hop is None:
        validated_rate_1hop = float(d["neighborhood_default_rate"])

    feature_values: OrderedDict[str, float | int] = OrderedDict()
    feature_trace_38: list[dict[str, Any]] = []

    def add_feature(
        feature_name: str,
        value: float | int,
        source_type: str,
        *,
        derived_from_fields: list[str] | None = None,
        derivation_formula: str | None = None,
        default_reason: str | None = None,
        median_value_used: float | None = None,
    ) -> None:
        casted_value = _cast_feature_value(feature_name, value)
        feature_values[feature_name] = casted_value
        feature_trace_38.append(
            _build_trace_row(
                index=len(feature_trace_38) + 1,
                feature_name=feature_name,
                value=casted_value,
                source_type=source_type,
                derived_from_fields=derived_from_fields,
                derivation_formula=derivation_formula,
                default_reason=default_reason,
                median_value_used=median_value_used,
            )
        )

    def add_median(feature_name: str, reason: str) -> None:
        median_value = _median_or_fallback(feature_name, training_medians)
        add_feature(
            feature_name,
            median_value,
            "training_set_median",
            default_reason=reason,
            median_value_used=float(median_value),
        )

    add_feature(
        "borrower_id",
        -1,
        "derived_from_form",
        derivation_formula="constant sentinel borrower_id=-1 for manual walk-in applicants",
    )

    add_median(
        "group_id",
        "No verified lending-group assignment available at manual application time.",
    )

    add_feature(
        "prior_CMR_usage",
        d["prior_CMR_usage"],
        "direct_from_form",
        derived_from_fields=["prior_CMR_usage"],
    )
    add_feature(
        "store_visit_count",
        d["store_visit_count"],
        "direct_from_form",
        derived_from_fields=["store_visit_count"],
    )
    add_feature(
        "CoDi_wallet_flag",
        d["CoDi_wallet_flag"],
        "direct_from_form",
        derived_from_fields=["CoDi_wallet_flag"],
    )
    add_feature(
        "INE_verified_flag",
        d["INE_verified_flag"],
        "direct_from_form",
        derived_from_fields=["INE_verified_flag"],
    )
    add_feature(
        "rural_flag", d["rural_flag"], "direct_from_form", derived_from_fields=["rural_flag"]
    )
    add_feature(
        "indigenous_proxy",
        d["indigenous_proxy"],
        "direct_from_form",
        derived_from_fields=["indigenous_proxy"],
    )
    add_feature("age", d["age"], "direct_from_form", derived_from_fields=["age"])

    add_median(
        "cycle_number",
        "Observed lending-cycle history is unavailable for new walk-in applicants.",
    )

    add_feature(
        "cohesion_score",
        d["group_cohesion_score"],
        "derived_from_form",
        derived_from_fields=["group_cohesion_score"],
        derivation_formula="cohesion_score := group_cohesion_score",
    )

    add_median("loan_count", "Historical loan count is unavailable for first-time manual scoring.")
    add_median(
        "avg_loan_amount_MXN",
        "Historical loan ledger is unavailable for first-time manual scoring.",
    )
    add_median(
        "max_loan_amount_MXN",
        "Historical loan ledger is unavailable for first-time manual scoring.",
    )
    add_median(
        "cmr_credit_line_share",
        "Historical CMR share is unavailable for first-time manual scoring.",
    )
    add_median(
        "oxxo_cash_backed_share",
        "Historical OXXO cash-backed share is unavailable for first-time manual scoring.",
    )
    add_median(
        "repayment_latency_days",
        "Repayment behavior history is unavailable for first-time manual scoring.",
    )
    add_median(
        "on_time_repayment_share",
        "Repayment behavior history is unavailable for first-time manual scoring.",
    )

    add_feature(
        "routine_score",
        d["call_routine_score"],
        "derived_from_form",
        derived_from_fields=["call_routine_score"],
        derivation_formula="routine_score := call_routine_score",
    )
    add_feature(
        "messaging_frequency",
        d["messaging_frequency"],
        "direct_from_form",
        derived_from_fields=["messaging_frequency"],
    )
    add_feature(
        "call_volume_stability",
        d["call_volume_stability"],
        "direct_from_form",
        derived_from_fields=["call_volume_stability"],
    )
    add_feature("app_opens", d["app_opens"], "direct_from_form", derived_from_fields=["app_opens"])
    add_feature(
        "location_variance",
        d["location_variance"],
        "direct_from_form",
        derived_from_fields=["location_variance"],
    )
    add_feature(
        "Falabella_app_session_flag",
        1 if int(d["app_opens"]) > 0 else 0,
        "derived_from_form",
        derived_from_fields=["app_opens"],
        derivation_formula="Falabella_app_session_flag := 1 if app_opens > 0 else 0",
    )
    add_median(
        "routine_entropy",
        "Longitudinal routine entropy requires historical mobile events and is not directly observable at intake.",
    )
    add_feature(
        "CoDi_transaction_regularity",
        0.5 if int(d["CoDi_wallet_flag"]) == 1 else 0.0,
        "derived_from_form",
        derived_from_fields=["CoDi_wallet_flag"],
        derivation_formula="CoDi_transaction_regularity := 0.5 if CoDi_wallet_flag=1 else 0.0",
    )
    add_feature(
        "falabella_app_session_recency",
        7 if int(d["app_opens"]) > 0 else 30,
        "derived_from_form",
        derived_from_fields=["app_opens"],
        derivation_formula="falabella_app_session_recency := 7 if app_opens > 0 else 30",
    )

    if graph_verified and validated_peer_count > 0:
        graph_reason = "Graph verification succeeded against known borrower nodes."
        degree = float(validated_peer_count) / 90000.0
        rate_1hop = float(validated_rate_1hop)
        rate_2hop = float(rate_1hop * 0.6)
        contagion = float(rate_1hop * float(validated_peer_count) * 0.1)
        community_flag = 1 if validated_peer_count >= 3 else 0

        add_feature(
            "degree_centrality",
            degree,
            "existing_profile",
            derived_from_fields=["graph_verification.linked_borrower_ids"],
            derivation_formula="degree_centrality := validated_peer_count / 90000",
            default_reason=graph_reason,
        )
        add_median(
            "weighted_tie_strength",
            "Tie-strength details require observed edge weights; median used even after peer verification.",
        )
        add_median(
            "betweenness_centrality",
            "Betweenness requires full network traversal context unavailable at intake.",
        )
        add_feature(
            "neighborhood_default_rate_1hop",
            rate_1hop,
            "existing_profile",
            derived_from_fields=["graph_verification.linked_borrower_ids"],
            derivation_formula="mean(default_flag) across validated linked borrower nodes",
            default_reason=graph_reason,
        )
        add_feature(
            "neighborhood_default_rate_2hop",
            rate_2hop,
            "existing_profile",
            derived_from_fields=["graph_verification.linked_borrower_ids"],
            derivation_formula="neighborhood_default_rate_2hop := neighborhood_default_rate_1hop * 0.6",
            default_reason=graph_reason,
        )
        add_median(
            "pagerank_score",
            "PageRank score requires global graph structure unavailable at intake.",
        )
        add_feature(
            "community_membership_flag",
            community_flag,
            "existing_profile",
            derived_from_fields=["graph_verification.linked_borrower_ids"],
            derivation_formula="community_membership_flag := 1 if validated_peer_count >= 3 else 0",
            default_reason=graph_reason,
        )
        add_median(
            "default_flag",
            "Outcome label is unknown at decision time; training median sentinel applied for schema parity.",
        )
        add_median(
            "sequential_lending_flag",
            "Sequential-lending history is unavailable for first-time walk-in applicants.",
        )
        add_feature(
            "peer_default_contagion_score",
            contagion,
            "existing_profile",
            derived_from_fields=["graph_verification.linked_borrower_ids"],
            derivation_formula=(
                "peer_default_contagion_score := neighborhood_default_rate_1hop * "
                "validated_peer_count * 0.1"
            ),
            default_reason=graph_reason,
        )
    else:
        add_median(
            "degree_centrality",
            "Graph linkage is unverified; conservative median used instead of self-reported peer count.",
        )
        add_median(
            "weighted_tie_strength",
            "Graph linkage is unverified; conservative median used for tie strength.",
        )
        add_median(
            "betweenness_centrality",
            "Graph linkage is unverified; conservative median used for broker-position metric.",
        )
        add_median(
            "neighborhood_default_rate_1hop",
            "Graph linkage is unverified; conservative median used for neighborhood default rate.",
        )
        add_median(
            "neighborhood_default_rate_2hop",
            "Graph linkage is unverified; conservative median used for extended neighborhood default rate.",
        )
        add_median(
            "pagerank_score",
            "Graph linkage is unverified; conservative median used for PageRank.",
        )
        add_median(
            "community_membership_flag",
            "Graph linkage is unverified; conservative median used for community indicator.",
        )
        add_median(
            "default_flag",
            "Outcome label is unknown at decision time; training median sentinel applied for schema parity.",
        )
        add_median(
            "sequential_lending_flag",
            "Sequential-lending history is unavailable for first-time walk-in applicants.",
        )
        add_median(
            "peer_default_contagion_score",
            "Graph linkage is unverified; conservative median used for contagion score.",
        )

    add_feature(
        "gender_female_flag",
        1 if d["gender"] == "Female" else 0,
        "derived_from_form",
        derived_from_fields=["gender"],
        derivation_formula="gender_female_flag := 1 if gender is Female else 0",
    )

    _assert_trace_integrity(feature_values, feature_trace_38)
    model_vector_38 = np.asarray([feature_values[c] for c in FEATURE_COLUMNS], dtype=float)
    feature_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
    return model_vector_38, feature_trace_38, feature_df


def build_existing_feature_bundle(
    borrower_id: int,
    features_table: pd.DataFrame,
) -> tuple[np.ndarray, list[dict[str, Any]], pd.DataFrame]:
    """Build canonical existing-borrower tabular vector plus full per-feature trace metadata."""

    feature_row = features_table.loc[features_table["borrower_id"] == borrower_id]
    if feature_row.empty:
        raise ValueError(f"borrower_id {borrower_id} not found in features_table")

    row = feature_row.iloc[0]
    feature_values: OrderedDict[str, float | int] = OrderedDict()
    trace_rows: list[dict[str, Any]] = []

    for idx, feature_name in enumerate(FEATURE_COLUMNS, start=1):
        value = row[feature_name]
        source_type = (
            "existing_profile" if feature_name == "borrower_id" else "existing_precomputed"
        )
        casted_value = _cast_feature_value(feature_name, value)
        feature_values[feature_name] = casted_value
        trace_rows.append(
            _build_trace_row(
                index=idx,
                feature_name=feature_name,
                value=casted_value,
                source_type=source_type,
                default_reason=(
                    "Loaded from processed features.parquet row for existing borrower profile."
                    if source_type == "existing_precomputed"
                    else "Borrower identifier from existing profile lookup."
                ),
            )
        )

    _assert_trace_integrity(feature_values, trace_rows)
    model_vector_38 = np.asarray([feature_values[c] for c in FEATURE_COLUMNS], dtype=float)
    feature_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
    return model_vector_38, trace_rows, feature_df


def build_feature_vector(input_data: BorrowerInput) -> pd.DataFrame:
    """Build a 1-row DataFrame in exact features.parquet order.

    Notes:
    - Some loan-officer form fields are not present in the current 38-column
      training table schema. They are retained in request validation but are
      not directly emitted as columns because the parquet schema is authoritative.
    - Inference alignment to model.feature_names_in_ is handled in api/main.py.
    """

    model_vector_38, _, feature_df = build_manual_feature_bundle(
        input_data=input_data,
        training_medians={},
        graph_verification=None,
    )
    if model_vector_38.shape[0] != 38:
        raise ValueError("Manual feature vector must have exactly 38 tabular features")
    return feature_df


def determine_cold_start_phase(
    input_data: BorrowerInput,
    graph_verified: bool = False,
    validated_peer_count: int = 0,
) -> int:
    if not graph_verified or validated_peer_count <= 0:
        if input_data.store_visit_count < 3 and input_data.CoDi_wallet_flag == 0:
            return 1
        return 2

    if (
        input_data.store_visit_count < 3
        and input_data.CoDi_wallet_flag == 0
        and validated_peer_count == 0
    ):
        return 1
    return 3


def compute_credit_line(phase: int, risk_score: float) -> int:
    phase_limits = {
        1: (500, 2000),
        2: (2000, 8000),
        3: (8000, 25000),
    }
    low, high = phase_limits[phase]
    return int(high - (high - low) * float(risk_score))


def build_feature_vector_from_existing(
    borrower_id: int,
    features_table: pd.DataFrame,
    embeddings_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For an existing borrower, pull pre-computed features and GNN embeddings
    directly from stored tables.

    Returns: (feature_vector, embedding_vector)
    """

    model_vector_38, _, _ = build_existing_feature_bundle(
        borrower_id=borrower_id, features_table=features_table
    )
    feature_vector = model_vector_38[1:]

    embedding_row = embeddings_table.loc[embeddings_table["borrower_id"] == borrower_id]
    if embedding_row.empty:
        raise ValueError(f"borrower_id {borrower_id} not found in embeddings_table")

    embedding_cols = sorted([c for c in embeddings_table.columns if c.startswith("emb_")])
    embedding_vector = embedding_row.iloc[0][embedding_cols].to_numpy(dtype=float)

    return feature_vector, embedding_vector


def determine_cold_start_phase_existing(profile: dict) -> int:
    """Determine borrower phase using observed existing-borrower data."""

    store_visits = int(profile.get("store_visit_count", 0))
    codi = int(profile.get("CoDi_wallet_flag", 0))
    peers = int(profile.get("peer_connection_count", 0))

    if store_visits < 3 and codi == 0 and peers == 0:
        return 1
    if peers == 0:
        return 2
    return 3
