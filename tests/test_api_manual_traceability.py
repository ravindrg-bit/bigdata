from __future__ import annotations

import math

import numpy as np
from fastapi.testclient import TestClient

from api import main
from api.feature_builder import (
    FEATURE_COLUMNS,
    build_manual_feature_bundle,
    determine_cold_start_phase,
)
from api.schemas import BorrowerInput, PredictionResponse

MANUAL_PAYLOAD = {
    "age": 24,
    "gender": "Female",
    "rural_flag": 0,
    "indigenous_proxy": 0,
    "INE_verified_flag": 1,
    "married_flag": 0,
    "num_children": 0,
    "monthly_income_MXN": 1471,
    "rent_MXN": 0,
    "informal_loans_flag": 0,
    "formal_debt_flag": 0,
    "electricity_water_MXN": 388,
    "store_visit_count": 9,
    "prior_CMR_usage": 0,
    "CoDi_wallet_flag": 0,
    "call_routine_score": 0.69,
    "messaging_frequency": 41,
    "call_volume_stability": 0.38,
    "app_opens": 29,
    "location_variance": 0.49,
    "peer_connections": 5,
    "group_cohesion_score": 0.33,
    "neighborhood_default_rate": 0.40,
}


def _debug_headers() -> dict[str, str]:
    if main.DEBUG_COMPARE_TOKEN:
        return {"X-Debug-Token": main.DEBUG_COMPARE_TOKEN}
    return {}


def test_feature_trace_integrity_has_38_and_source_types() -> None:
    payload = BorrowerInput(**MANUAL_PAYLOAD)
    graph_context = main._resolve_manual_graph_context(payload)

    model_vector_38, trace_38, feature_df_38 = build_manual_feature_bundle(
        input_data=payload,
        training_medians=main.FEATURE_MEDIANS,
        graph_verification=graph_context,
    )

    assert model_vector_38.shape == (38,)
    assert len(trace_38) == 38
    assert list(feature_df_38.columns) == FEATURE_COLUMNS
    assert feature_df_38.shape == (1, 38)

    for row in trace_38:
        assert row["source_type"]


def test_manual_unverified_defaults_to_zero_embeddings_and_conservative_phase() -> None:
    payload = BorrowerInput(**MANUAL_PAYLOAD)
    graph_context = main._resolve_manual_graph_context(payload)

    embedding_vector = main._build_manual_embedding_vector(graph_context)
    phase = determine_cold_start_phase(
        payload,
        graph_verified=bool(graph_context.get("verified", False)),
        validated_peer_count=int(graph_context.get("validated_peer_count", 0) or 0),
    )

    assert graph_context["verified"] is False
    assert embedding_vector.shape == (main.EMBEDDING_DIMENSION,)
    assert bool(np.allclose(embedding_vector, 0.0)) is True
    assert phase in {1, 2}
    assert phase != 3


def test_existing_endpoint_response_is_schema_compatible() -> None:
    client = TestClient(main.app)
    response = client.post("/predict/existing/13021")

    assert response.status_code == 200
    payload = response.json()

    required_keys = {
        "raw_score",
        "risk_percentile",
        "risk_band",
        "risk_score",
        "decision",
        "cold_start_phase",
        "credit_line_MXN",
        "top_drivers",
        "explanation",
        "borrower_id",
        "actual_outcome",
        "model_correct",
    }
    assert required_keys.issubset(set(payload.keys()))

    validated = PredictionResponse.model_validate(payload)
    assert validated.borrower_id == 13021


def test_debug_comparison_endpoint_regression_13021_complete_and_finite() -> None:
    client = TestClient(main.app)
    response = client.post(
        "/debug/compare/existing-vs-manual/13021",
        json=MANUAL_PAYLOAD,
        headers=_debug_headers(),
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["feature_mapping_38"]) == 38

    for row in data["feature_mapping_38"]:
        assert row["feature_name"]
        assert row["manual_source_type"]
        assert math.isfinite(float(row["existing_value"]))
        assert math.isfinite(float(row["manual_value"]))
        assert math.isfinite(float(row["delta"]))

    assert data["source_counts"]["direct"] >= 1
    assert data["source_counts"]["derived"] >= 1
    assert data["source_counts"]["median_default"] >= 1

    existing_embedding_diag = data["embedding_diagnostics"]["existing"]
    manual_embedding_diag = data["embedding_diagnostics"]["manual"]
    assert int(existing_embedding_diag["dimension"]) == main.EMBEDDING_DIMENSION
    assert int(manual_embedding_diag["dimension"]) == main.EMBEDDING_DIMENSION
    assert manual_embedding_diag["zero_vector"] is True

    existing_top5 = data["shap_comparison"]["existing_top5"]
    manual_top5 = data["shap_comparison"]["manual_top5"]
    assert len(existing_top5) == 5
    assert len(manual_top5) == 5

    for item in existing_top5 + manual_top5:
        assert item["feature"]
        assert math.isfinite(float(item["shap_contribution"]))
