from __future__ import annotations

import os
import random
from collections import Counter
from typing import Any
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware

try:
    from .data_loader import BORROWER_ID_LIST, BORROWER_PROFILES, EMBEDDINGS_TABLE, FEATURES_TABLE
    from .feature_builder import (
        FEATURE_COLUMNS,
        build_existing_feature_bundle,
        build_feature_vector_from_existing,
        build_manual_feature_bundle,
        compute_credit_line,
        compute_training_feature_medians,
        determine_cold_start_phase,
        determine_cold_start_phase_existing,
    )
    from .model_loader import (
        get_explainer,
        get_feature_label,
        get_graph_signal_label,
        get_model_info,
        get_model,
        percentile_to_decision,
        score_to_percentile,
    )
    from .schemas import (
        BorrowerInput,
        BorrowerListResponse,
        BorrowerProfile,
        BorrowerSearchResult,
        PredictionResponse,
        TopDriver,
    )
except ImportError:
    if __package__:
        raise
    # Supports local invocation from the api/ directory: uvicorn main:app
    from data_loader import BORROWER_ID_LIST, BORROWER_PROFILES, EMBEDDINGS_TABLE, FEATURES_TABLE
    from feature_builder import (
        FEATURE_COLUMNS,
        build_existing_feature_bundle,
        build_feature_vector_from_existing,
        build_manual_feature_bundle,
        compute_credit_line,
        compute_training_feature_medians,
        determine_cold_start_phase,
        determine_cold_start_phase_existing,
    )
    from model_loader import (
        get_explainer,
        get_feature_label,
        get_graph_signal_label,
        get_model_info,
        get_model,
        percentile_to_decision,
        score_to_percentile,
    )
    from schemas import (
        BorrowerInput,
        BorrowerListResponse,
        BorrowerProfile,
        BorrowerSearchResult,
        PredictionResponse,
        TopDriver,
    )


app = FastAPI(
    title="Falabella Risk Engine API",
    description="Hybrid credit risk scoring for thin-file borrowers in Mexico",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://falabella-mexico-credits.lovable.app",
        "https://bancoconcept.xyz",
        "https://www.bancoconcept.xyz",
    ],
    allow_origin_regex=r"https://.*\.lovable(project)?\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BORROWER_ID_SET = set(BORROWER_ID_LIST)
MIN_BORROWER_ID = BORROWER_ID_LIST[0] if BORROWER_ID_LIST else 0
MAX_BORROWER_ID = BORROWER_ID_LIST[-1] if BORROWER_ID_LIST else 0

BORROWER_PROFILE_TABLE = BORROWER_PROFILES.set_index("borrower_id", drop=False)
DEFAULTED_IDS = (
    BORROWER_PROFILE_TABLE.loc[BORROWER_PROFILE_TABLE["default_flag"] == 1, "borrower_id"]
    .astype(int)
    .tolist()
)
REPAID_IDS = (
    BORROWER_PROFILE_TABLE.loc[BORROWER_PROFILE_TABLE["default_flag"] == 0, "borrower_id"]
    .astype(int)
    .tolist()
)

EXISTING_EMBEDDING_COLUMNS = sorted([c for c in EMBEDDINGS_TABLE.columns if c.startswith("emb_")])
EMBEDDING_DIMENSION = len(EXISTING_EMBEDDING_COLUMNS)
EMBEDDING_TABLE_BY_ID = EMBEDDINGS_TABLE.set_index("borrower_id", drop=False)

FEATURE_MEDIANS = compute_training_feature_medians(FEATURES_TABLE)
FEATURE_SCHEMA_VERSION = "features.parquet.v1.38"
MANUAL_SCORING_POLICY_VERSION = "manual-cold-start.v2"
MANUAL_ZERO_EMBEDDINGS_DEFAULT = True

DEBUG_COMPARE_TOKEN = os.getenv("DEBUG_COMPARE_TOKEN")


ENDPOINT_DESCRIPTIONS = [
    {"method": "GET", "path": "/health", "description": "Health check"},
    {
        "method": "GET",
        "path": "/model-info",
        "description": "Scoring method and decision thresholds",
    },
    {"method": "GET", "path": "/stats", "description": "Dataset summary stats"},
    {"method": "GET", "path": "/schema", "description": "API schema and endpoint docs"},
    {"method": "GET", "path": "/borrower/random", "description": "Random borrower profile"},
    {
        "method": "GET",
        "path": "/borrower/search",
        "description": "Search by borrower ID prefix or city",
    },
    {
        "method": "GET",
        "path": "/borrower/{borrower_id}",
        "description": "Lookup specific borrower profile",
    },
    {
        "method": "POST",
        "path": "/predict/existing/{borrower_id}",
        "description": "Score existing borrower using pre-computed features and embeddings",
    },
    {
        "method": "POST",
        "path": "/predict/manual",
        "description": "Score manual borrower input using estimated defaults and zero embeddings",
    },
    {
        "method": "POST",
        "path": "/debug/compare/existing-vs-manual/{borrower_id}",
        "description": "Internal diagnostic comparison for existing-vs-manual scoring traces",
    },
]


def _extract_shap_vector(shap_output: Any) -> np.ndarray:
    if isinstance(shap_output, list):
        arr = np.asarray(shap_output[-1], dtype=float)
        return arr[0] if arr.ndim > 1 else arr

    if hasattr(shap_output, "values"):
        arr = np.asarray(shap_output.values, dtype=float)
        if arr.ndim == 3:
            return arr[0, :, -1]
        if arr.ndim == 2:
            return arr[0]
        return arr

    arr = np.asarray(shap_output, dtype=float)
    if arr.ndim == 3:
        return arr[0, :, -1]
    if arr.ndim == 2:
        return arr[0]
    return arr


def _to_native(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_borrower_id(borrower_id: int) -> None:
    if borrower_id not in BORROWER_ID_SET:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Borrower ID {borrower_id} not found. "
                f"Valid range: {MIN_BORROWER_ID} to {MAX_BORROWER_ID}"
            ),
        )


def _get_profile_row(borrower_id: int) -> pd.Series:
    return BORROWER_PROFILE_TABLE.loc[borrower_id]


def _row_to_profile(row: pd.Series) -> BorrowerProfile:
    payload: dict[str, Any] = {}
    for field_name in BorrowerProfile.model_fields:
        if field_name == "actual_outcome":
            continue
        payload[field_name] = _to_native(row.get(field_name))

    default_flag = int(payload.get("default_flag", 0) or 0)
    payload["actual_outcome"] = "defaulted" if default_flag == 1 else "repaid"
    return BorrowerProfile(**payload)


def _align_to_model_features(raw_input: pd.DataFrame) -> pd.DataFrame:
    model = get_model()
    model_features = list(getattr(model, "feature_names_in_", []))
    if not model_features:
        return raw_input.astype(float)

    x = raw_input.copy()
    return x.reindex(columns=model_features, fill_value=0.0).astype(float)


def _build_model_input(feature_df_38: pd.DataFrame, embedding_vector: np.ndarray) -> pd.DataFrame:
    if int(np.asarray(embedding_vector).shape[0]) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding vector length mismatch: expected {EMBEDDING_DIMENSION}, "
            f"got {int(np.asarray(embedding_vector).shape[0])}"
        )

    x = feature_df_38.drop(columns=["borrower_id"], errors="ignore").copy()
    embedding_df = pd.DataFrame(
        [np.asarray(embedding_vector, dtype=float)],
        columns=EXISTING_EMBEDDING_COLUMNS,
    )
    x = pd.concat([x, embedding_df], axis=1)
    return _align_to_model_features(x)


def _resolve_manual_graph_context(payload: BorrowerInput) -> dict[str, Any]:
    context: dict[str, Any] = {
        "verified": False,
        "validation_status": "unverified",
        "validation_note": (
            "No graph verification metadata provided; self-reported peer fields are treated "
            "as unverified for phase promotion and embedding generation."
        ),
        "validated_peer_ids": [],
        "invalid_peer_ids": [],
        "validated_peer_count": 0,
        "validated_neighborhood_default_rate_1hop": None,
        "verification_method": None,
        "verifier_reference": None,
    }

    metadata = payload.graph_verification
    if metadata is None:
        return context

    candidate_ids = sorted({int(bid) for bid in metadata.linked_borrower_ids})
    invalid_ids = [bid for bid in candidate_ids if bid not in BORROWER_ID_SET]

    context["verification_method"] = metadata.verification_method
    context["verifier_reference"] = metadata.verifier_reference
    context["invalid_peer_ids"] = invalid_ids

    if invalid_ids:
        context["validation_status"] = "invalid"
        context["validation_note"] = (
            "Graph verification failed because some linked_borrower_ids were not found in "
            "known borrower nodes. Conservative manual policy was applied."
        )
        return context

    if not candidate_ids:
        context["validation_status"] = "invalid"
        context["validation_note"] = (
            "Graph verification failed because no valid linked_borrower_ids were provided. "
            "Conservative manual policy was applied."
        )
        return context

    observed_defaults = (
        BORROWER_PROFILE_TABLE.loc[candidate_ids, "default_flag"]
        .astype(float)
        .to_numpy(dtype=float)
    )
    neighborhood_default_rate_1hop = (
        float(np.mean(observed_defaults)) if observed_defaults.size else 0.0
    )

    context.update(
        {
            "verified": True,
            "validation_status": "verified",
            "validation_note": (
                "Graph verification succeeded against known borrower nodes; graph-dependent "
                "promotion and embedding proxy are enabled."
            ),
            "validated_peer_ids": candidate_ids,
            "validated_peer_count": int(len(candidate_ids)),
            "validated_neighborhood_default_rate_1hop": neighborhood_default_rate_1hop,
        }
    )
    return context


def _build_manual_embedding_vector(graph_context: dict[str, Any]) -> np.ndarray:
    if not graph_context.get("verified", False):
        return np.zeros(EMBEDDING_DIMENSION, dtype=float)

    linked_ids = [int(x) for x in graph_context.get("validated_peer_ids", [])]
    if not linked_ids:
        return np.zeros(EMBEDDING_DIMENSION, dtype=float)

    try:
        rows = EMBEDDING_TABLE_BY_ID.loc[linked_ids, EXISTING_EMBEDDING_COLUMNS]
    except KeyError:
        return np.zeros(EMBEDDING_DIMENSION, dtype=float)

    values = np.asarray(rows.to_numpy(dtype=float), dtype=float)
    if values.size == 0:
        return np.zeros(EMBEDDING_DIMENSION, dtype=float)

    mean_embedding = values.mean(axis=0)
    if mean_embedding.ndim != 1 or int(mean_embedding.shape[0]) != EMBEDDING_DIMENSION:
        return np.zeros(EMBEDDING_DIMENSION, dtype=float)

    if not np.all(np.isfinite(mean_embedding)):
        return np.zeros(EMBEDDING_DIMENSION, dtype=float)

    return mean_embedding.astype(float)


def _embedding_diagnostics(vector: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(vector, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    return {
        "dimension": int(arr.shape[0]),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "norm": float(np.linalg.norm(arr)),
        "zero_vector": bool(np.allclose(arr, 0.0)),
    }


def _to_ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _build_explanation_sentence(
    tabular_drivers: list[TopDriver],
    risk_percentile: float,
    risk_band: str,
) -> str:
    percentile_value = int(round(float(risk_percentile) * 100))
    ordinal = _to_ordinal(percentile_value)
    band_sentence = (
        f"This applicant's risk is at the {ordinal} percentile - "
        f"higher risk than {percentile_value}% of the borrower population "
        f"({risk_band} band)."
    )

    if not tabular_drivers:
        return band_sentence

    if len(tabular_drivers) == 1:
        d1 = tabular_drivers[0]
        return (
            f"{band_sentence} "
            f"Primary factor: {d1.feature} {d1.direction} "
            f"(SHAP {d1.shap_contribution:+.4f})."
        )

    d1, d2 = tabular_drivers[0], tabular_drivers[1]
    return (
        f"{band_sentence} "
        f"Primary factors: {d1.feature} {d1.direction} (SHAP {d1.shap_contribution:+.4f}) "
        f"and {d2.feature} {d2.direction} (SHAP {d2.shap_contribution:+.4f})."
    )


def _score_model_input(
    model_input: pd.DataFrame,
    return_contrib: bool = False,
) -> (
    tuple[float, list[TopDriver], list[TopDriver]]
    | tuple[float, list[TopDriver], list[TopDriver], pd.DataFrame]
):
    model = get_model()
    explainer = get_explainer()

    raw_score = float(model.predict_proba(model_input)[0, 1])

    shap_output = explainer.shap_values(model_input)
    shap_vector = _extract_shap_vector(shap_output)

    contrib = pd.DataFrame(
        {
            "feature": [str(col) for col in model_input.columns],
            "value": model_input.iloc[0].to_numpy(dtype=float),
            "shap_contribution": np.asarray(shap_vector, dtype=float),
        }
    )
    contrib["is_graph_embedding"] = contrib["feature"].str.startswith("emb_")

    tabular_contrib = contrib.loc[~contrib["is_graph_embedding"]].copy()
    tabular_contrib["abs_shap"] = tabular_contrib["shap_contribution"].abs()
    top5_tabular = tabular_contrib.sort_values("abs_shap", ascending=False).head(5)

    graph_contrib = contrib.loc[contrib["is_graph_embedding"]]
    graph_total = (
        float(graph_contrib["shap_contribution"].sum()) if not graph_contrib.empty else 0.0
    )

    tabular_drivers: list[TopDriver] = []
    for _, row in top5_tabular.iterrows():
        direction = "increased risk" if float(row["shap_contribution"]) >= 0 else "reduced risk"
        tabular_drivers.append(
            TopDriver(
                feature=get_feature_label(str(row["feature"])),
                value=float(row["value"]),
                shap_contribution=float(row["shap_contribution"]),
                direction=direction,
            )
        )

    tabular_drivers = sorted(
        tabular_drivers,
        key=lambda d: abs(float(d.shap_contribution)),
        reverse=True,
    )

    top_drivers = tabular_drivers.copy()

    if abs(graph_total) > 0.05:
        graph_direction = "increased risk" if graph_total >= 0 else "reduced risk"
        top_drivers.append(
            TopDriver(
                feature=get_graph_signal_label(),
                value=None,
                shap_contribution=graph_total,
                direction=graph_direction,
            )
        )

    top_drivers = sorted(
        top_drivers,
        key=lambda d: abs(float(d.shap_contribution)),
        reverse=True,
    )[:6]

    if return_contrib:
        return raw_score, top_drivers, tabular_drivers, contrib

    return raw_score, top_drivers, tabular_drivers


def _serialize_top_drivers(drivers: list[TopDriver], limit: int = 5) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for driver in drivers[:limit]:
        payload = driver.model_dump()
        payload["shap_contribution"] = float(payload["shap_contribution"])
        payload["value"] = None if payload["value"] is None else float(payload["value"])
        serialized.append(payload)
    return serialized


def _prediction_snapshot(
    raw_score: float,
    percentile: float,
    risk_band: str,
    decision: str,
    credit_line: int,
    phase: int,
) -> dict[str, Any]:
    return {
        "raw_score": float(raw_score),
        "risk_percentile": float(percentile),
        "risk_band": risk_band,
        "decision": decision,
        "credit_line_MXN": int(credit_line),
        "phase": int(phase),
    }


def _authorize_debug_request(x_debug_token: str | None) -> None:
    if not DEBUG_COMPARE_TOKEN:
        return
    if x_debug_token != DEBUG_COMPARE_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid X-Debug-Token for debug endpoint")


@app.post("/predict/manual", response_model=PredictionResponse)
def predict_manual(payload: BorrowerInput) -> PredictionResponse:
    graph_context = _resolve_manual_graph_context(payload)
    _, feature_trace_38, feature_df_38 = build_manual_feature_bundle(
        input_data=payload,
        training_medians=FEATURE_MEDIANS,
        graph_verification=graph_context,
    )

    if len(feature_trace_38) != 38:
        raise HTTPException(status_code=500, detail="Manual feature trace integrity check failed")

    phase = determine_cold_start_phase(
        payload,
        graph_verified=bool(graph_context.get("verified", False)),
        validated_peer_count=int(graph_context.get("validated_peer_count", 0) or 0),
    )
    embedding_vector = _build_manual_embedding_vector(graph_context)
    model_input = _build_model_input(feature_df_38, embedding_vector)

    raw_score, top_drivers, tabular_drivers = _score_model_input(model_input)
    risk_percentile = score_to_percentile(raw_score)
    decision, risk_band = percentile_to_decision(risk_percentile)
    credit_line = compute_credit_line(phase, risk_percentile)
    explanation = _build_explanation_sentence(tabular_drivers, risk_percentile, risk_band)

    return PredictionResponse(
        raw_score=raw_score,
        risk_percentile=risk_percentile,
        risk_band=risk_band,
        risk_score=risk_percentile,
        decision=decision,
        cold_start_phase=phase,
        credit_line_MXN=credit_line,
        top_drivers=top_drivers,
        explanation=explanation,
        borrower_id=None,
        actual_outcome=None,
        model_correct=None,
    )


@app.post("/predict/existing/{borrower_id}", response_model=PredictionResponse)
def predict_existing(borrower_id: int) -> PredictionResponse:
    _require_borrower_id(borrower_id)
    row = _get_profile_row(borrower_id)

    _, existing_trace_38, existing_feature_df_38 = build_existing_feature_bundle(
        borrower_id=borrower_id,
        features_table=FEATURES_TABLE,
    )
    if len(existing_trace_38) != 38:
        raise HTTPException(status_code=500, detail="Existing feature trace integrity check failed")

    _, embedding_vector = build_feature_vector_from_existing(
        borrower_id=borrower_id,
        features_table=FEATURES_TABLE,
        embeddings_table=EMBEDDINGS_TABLE,
    )
    model_input = _build_model_input(existing_feature_df_38, embedding_vector)

    raw_score, top_drivers, tabular_drivers = _score_model_input(model_input)
    risk_percentile = score_to_percentile(raw_score)
    decision, risk_band = percentile_to_decision(risk_percentile)
    explanation = _build_explanation_sentence(tabular_drivers, risk_percentile, risk_band)

    phase = determine_cold_start_phase_existing(row.to_dict())
    credit_line = compute_credit_line(phase, risk_percentile)

    default_flag = int(_to_native(row.get("default_flag")) or 0)
    actual_outcome = "defaulted" if default_flag == 1 else "repaid"
    model_correct = bool(
        (decision != "decline" and default_flag == 0)
        or (decision == "decline" and default_flag == 1)
    )

    return PredictionResponse(
        raw_score=raw_score,
        risk_percentile=risk_percentile,
        risk_band=risk_band,
        risk_score=risk_percentile,
        decision=decision,
        cold_start_phase=phase,
        credit_line_MXN=credit_line,
        top_drivers=top_drivers,
        explanation=explanation,
        borrower_id=borrower_id,
        actual_outcome=actual_outcome,
        model_correct=model_correct,
    )


@app.post("/debug/compare/existing-vs-manual/{borrower_id}")
def debug_compare_existing_vs_manual(
    borrower_id: int,
    payload: BorrowerInput,
    x_debug_token: str | None = Header(default=None, alias="X-Debug-Token"),
) -> dict[str, Any]:
    _authorize_debug_request(x_debug_token)
    _require_borrower_id(borrower_id)

    existing_profile = _get_profile_row(borrower_id)
    existing_vector_38, _, existing_feature_df_38 = build_existing_feature_bundle(
        borrower_id=borrower_id,
        features_table=FEATURES_TABLE,
    )
    _, existing_embedding_vector = build_feature_vector_from_existing(
        borrower_id=borrower_id,
        features_table=FEATURES_TABLE,
        embeddings_table=EMBEDDINGS_TABLE,
    )

    manual_graph_context = _resolve_manual_graph_context(payload)
    manual_vector_38, manual_trace_38, manual_feature_df_38 = build_manual_feature_bundle(
        input_data=payload,
        training_medians=FEATURE_MEDIANS,
        graph_verification=manual_graph_context,
    )
    manual_embedding_vector = _build_manual_embedding_vector(manual_graph_context)

    existing_model_input = _build_model_input(existing_feature_df_38, existing_embedding_vector)
    manual_model_input = _build_model_input(manual_feature_df_38, manual_embedding_vector)

    existing_raw, existing_top, _, _ = _score_model_input(existing_model_input, return_contrib=True)
    manual_raw, manual_top, _, _ = _score_model_input(manual_model_input, return_contrib=True)

    existing_percentile = score_to_percentile(existing_raw)
    manual_percentile = score_to_percentile(manual_raw)

    existing_decision, existing_risk_band = percentile_to_decision(existing_percentile)
    manual_decision, manual_risk_band = percentile_to_decision(manual_percentile)

    existing_phase = determine_cold_start_phase_existing(existing_profile.to_dict())
    manual_phase = determine_cold_start_phase(
        payload,
        graph_verified=bool(manual_graph_context.get("verified", False)),
        validated_peer_count=int(manual_graph_context.get("validated_peer_count", 0) or 0),
    )

    existing_credit_line = compute_credit_line(existing_phase, existing_percentile)
    manual_credit_line = compute_credit_line(manual_phase, manual_percentile)

    feature_mapping_38: list[dict[str, Any]] = []
    for index, feature_name in enumerate(FEATURE_COLUMNS, start=1):
        existing_value = float(existing_vector_38[index - 1])
        manual_value = float(manual_vector_38[index - 1])
        delta = manual_value - existing_value
        manual_trace = manual_trace_38[index - 1]

        feature_mapping_38.append(
            {
                "index": index,
                "feature_name": feature_name,
                "existing_value": existing_value,
                "manual_value": manual_value,
                "match": bool(np.isclose(existing_value, manual_value, rtol=1e-6, atol=1e-9)),
                "delta": float(delta),
                "manual_source_type": manual_trace["source_type"],
                "manual_derived_from_fields": manual_trace.get("derived_from_fields", []),
                "manual_derivation_formula": manual_trace.get("derivation_formula"),
                "manual_default_reason": manual_trace.get("default_reason"),
                "manual_median_value_used": manual_trace.get("median_value_used"),
            }
        )

    source_counter = Counter(row["source_type"] for row in manual_trace_38)
    source_counts = {
        "direct": int(source_counter.get("direct_from_form", 0)),
        "derived": int(source_counter.get("derived_from_form", 0)),
        "median_default": int(source_counter.get("training_set_median", 0)),
        "existing_profile": int(source_counter.get("existing_profile", 0)),
    }

    existing_top5 = _serialize_top_drivers(existing_top, limit=5)
    manual_top5 = _serialize_top_drivers(manual_top, limit=5)

    existing_feature_set = {item["feature"] for item in existing_top5}
    manual_feature_set = {item["feature"] for item in manual_top5}
    overlap_features = sorted(existing_feature_set & manual_feature_set)

    divergence_notes: list[str] = []
    if bool(np.allclose(manual_embedding_vector, 0.0)):
        divergence_notes.append(
            "Manual embedding vector is all zeros under conservative cold-start policy "
            "because graph verification did not succeed."
        )
    if not overlap_features:
        divergence_notes.append(
            "Top SHAP drivers do not overlap between existing and manual paths."
        )
    else:
        divergence_notes.append(
            f"Top SHAP overlap count={len(overlap_features)} ({', '.join(overlap_features)})."
        )

    return {
        "borrower_id": borrower_id,
        "manual_graph_verification": manual_graph_context,
        "feature_mapping_38": feature_mapping_38,
        "source_counts": source_counts,
        "embedding_diagnostics": {
            "existing": _embedding_diagnostics(existing_embedding_vector),
            "manual": _embedding_diagnostics(manual_embedding_vector),
        },
        "prediction_comparison": {
            "existing": _prediction_snapshot(
                raw_score=existing_raw,
                percentile=existing_percentile,
                risk_band=existing_risk_band,
                decision=existing_decision,
                credit_line=existing_credit_line,
                phase=existing_phase,
            ),
            "manual": _prediction_snapshot(
                raw_score=manual_raw,
                percentile=manual_percentile,
                risk_band=manual_risk_band,
                decision=manual_decision,
                credit_line=manual_credit_line,
                phase=manual_phase,
            ),
        },
        "shap_comparison": {
            "existing_top5": existing_top5,
            "manual_top5": manual_top5,
            "overlap_features": overlap_features,
            "existing_only_features": sorted(existing_feature_set - manual_feature_set),
            "manual_only_features": sorted(manual_feature_set - existing_feature_set),
            "divergence_notes": divergence_notes,
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = get_model() is not None
    return {"status": "healthy", "model_loaded": loaded}


@app.get("/model-info")
def model_info() -> dict[str, float | int | str | bool]:
    return get_model_info(
        embedding_dimension=EMBEDDING_DIMENSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        manual_policy_version=MANUAL_SCORING_POLICY_VERSION,
        manual_zero_embeddings_default=MANUAL_ZERO_EMBEDDINGS_DEFAULT,
    )


@app.get("/stats")
def stats() -> dict[str, Any]:
    total_borrowers = int(len(BORROWER_PROFILES))
    total_defaulters = int(BORROWER_PROFILES["default_flag"].sum())
    total_repaid = int(total_borrowers - total_defaulters)

    default_rate = float(total_defaulters / total_borrowers) if total_borrowers else 0.0
    gender_norm = BORROWER_PROFILES["gender"].astype(str).str.strip().str.upper()
    female_share = float(gender_norm.isin(["F", "FEMALE"]).mean())
    rural_share = float((BORROWER_PROFILES["rural_flag"] == 1).mean())

    return {
        "total_borrowers": total_borrowers,
        "default_rate": default_rate,
        "total_defaulters": total_defaulters,
        "total_repaid": total_repaid,
        "mean_age": float(BORROWER_PROFILES["age"].mean()),
        "female_share": female_share,
        "rural_share": rural_share,
        "mean_income_MXN": float(BORROWER_PROFILES["monthly_income_MXN"].mean()),
        "mean_loan_count": float(BORROWER_PROFILES["loan_count"].mean()),
        "mean_risk_score": None,
    }


@app.get("/borrower/random", response_model=BorrowerProfile)
def borrower_random(
    default_status: Literal["defaulted", "repaid", "any"] = Query("any"),
) -> BorrowerProfile:
    if default_status == "defaulted":
        candidate_ids = DEFAULTED_IDS
    elif default_status == "repaid":
        candidate_ids = REPAID_IDS
    else:
        candidate_ids = BORROWER_ID_LIST

    if not candidate_ids:
        raise HTTPException(
            status_code=404, detail=f"No borrowers available for default_status={default_status}"
        )

    selected_id = int(random.choice(candidate_ids))
    row = _get_profile_row(selected_id)
    return _row_to_profile(row)


@app.get("/borrower/search", response_model=BorrowerListResponse)
def borrower_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
) -> BorrowerListResponse:
    query = q.strip()
    if not query:
        return BorrowerListResponse(total_borrowers=0, results=[])

    if query.isdigit():
        id_strings = BORROWER_PROFILES["borrower_id"].astype(str)
        matches = BORROWER_PROFILES.loc[id_strings.str.startswith(query)]
    else:
        city_strings = BORROWER_PROFILES["city"].fillna("").astype(str)
        matches = BORROWER_PROFILES.loc[city_strings.str.contains(query, case=False, regex=False)]

    matches = matches.sort_values("borrower_id")
    total_matches = int(len(matches))

    results: list[BorrowerSearchResult] = []
    for _, row in matches.head(limit).iterrows():
        results.append(
            BorrowerSearchResult(
                borrower_id=int(_to_native(row["borrower_id"])),
                age=int(_to_native(row["age"])),
                gender=str(_to_native(row["gender"])),
                city=str(_to_native(row["city"])),
                rural_flag=int(_to_native(row["rural_flag"])),
                loan_count=int(_to_native(row["loan_count"])),
                default_flag=int(_to_native(row["default_flag"])),
            )
        )

    return BorrowerListResponse(total_borrowers=total_matches, results=results)


@app.get("/borrower/{borrower_id}", response_model=BorrowerProfile)
def borrower_lookup(borrower_id: int) -> BorrowerProfile:
    _require_borrower_id(borrower_id)
    row = _get_profile_row(borrower_id)
    return _row_to_profile(row)


@app.get("/schema")
def schema() -> dict[str, Any]:
    model_schema = BorrowerInput.model_json_schema()
    properties = model_schema.get("properties", {})

    fields = []
    for name, spec in properties.items():
        fields.append(
            {
                "name": name,
                "type": spec.get("type", "string"),
                "enum": spec.get("enum"),
                "minimum": spec.get("minimum"),
                "maximum": spec.get("maximum"),
                "description": spec.get("description", ""),
            }
        )

    manual_unobservable_features = [
        "group_id",
        "cycle_number",
        "loan_count",
        "avg_loan_amount_MXN",
        "max_loan_amount_MXN",
        "cmr_credit_line_share",
        "oxxo_cash_backed_share",
        "repayment_latency_days",
        "on_time_repayment_share",
        "routine_entropy",
        "weighted_tie_strength",
        "betweenness_centrality",
        "pagerank_score",
        "default_flag",
        "sequential_lending_flag",
    ]

    graph_sensitive_features = [
        "degree_centrality",
        "neighborhood_default_rate_1hop",
        "neighborhood_default_rate_2hop",
        "community_membership_flag",
        "peer_default_contagion_score",
    ]

    return {
        "title": "BorrowerInput",
        "fields": fields,
        "endpoints": ENDPOINT_DESCRIPTIONS,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "manual_scoring_policy": {
            "version": MANUAL_SCORING_POLICY_VERSION,
            "zero_embeddings_default": MANUAL_ZERO_EMBEDDINGS_DEFAULT,
            "graph_verification_required_for_phase3": True,
            "policy_notes": [
                "Manual walk-in scoring uses estimated defaults (training medians) for unavailable historical fields.",
                "Self-reported peer_connections and neighborhood values are unverified inputs unless graph verification succeeds.",
                "Without successful graph verification, manual scoring stays in conservative cold-start tiers and uses zero graph embeddings.",
            ],
        },
        "manual_unobservable_features": manual_unobservable_features,
        "manual_graph_sensitive_features": graph_sensitive_features,
        "trace_schema": {
            "columns": [
                "index",
                "feature_name",
                "value",
                "source_type",
                "derived_from_fields",
                "derivation_formula",
                "default_reason",
                "median_value_used",
            ],
            "source_type_values": [
                "direct_from_form",
                "derived_from_form",
                "training_set_median",
                "existing_profile",
                "existing_precomputed",
            ],
        },
    }
