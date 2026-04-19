from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from feature_builder import build_feature_vector, compute_credit_line, determine_cold_start_phase
from model_loader import get_decision_threshold, get_explainer, get_feature_label, get_model
from schemas import BorrowerInput, PredictionResponse, TopDriver


app = FastAPI(
    title="Falabella Risk Engine API",
    description="Hybrid credit risk scoring for thin-file borrowers in Mexico",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to Lovable domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _build_model_input(feature_df: pd.DataFrame, phase: int) -> pd.DataFrame:
    model = get_model()
    model_features = list(getattr(model, "feature_names_in_", []))

    x = feature_df.drop(columns=["borrower_id", "default_flag"], errors="ignore").copy()

    # For phase 3 the API explicitly appends GNN embeddings. For new borrowers,
    # embeddings are zeros (cold-start behavior).
    if phase == 3:
        for idx in range(256):
            col = f"emb_{idx:03d}"
            if col not in x.columns:
                x[col] = 0.0

    # If phase 1/2, we still use the hybrid model and let missing embedding
    # columns be zero-filled during feature alignment.
    for col in model_features:
        if col not in x.columns:
            x[col] = 0.0

    return x[model_features].astype(float)


def _build_explanation_sentence(top_drivers: list[TopDriver]) -> str:
    if not top_drivers:
        return "No dominant risk drivers were identified for this prediction."

    if len(top_drivers) == 1:
        d1 = top_drivers[0]
        return (
            f"Primary factor: {d1.feature} {d1.direction} "
            f"(SHAP {d1.shap_contribution:+.4f})."
        )

    d1, d2 = top_drivers[0], top_drivers[1]
    return (
        f"Primary factors: {d1.feature} {d1.direction} (SHAP {d1.shap_contribution:+.4f}) "
        f"and {d2.feature} {d2.direction} (SHAP {d2.shap_contribution:+.4f})."
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: BorrowerInput) -> PredictionResponse:
    model = get_model()
    explainer = get_explainer()
    decision_threshold = get_decision_threshold()

    feature_df = build_feature_vector(payload)
    phase = determine_cold_start_phase(payload)
    model_input = _build_model_input(feature_df, phase)

    risk_score = float(model.predict_proba(model_input)[0, 1])
    decision = "approve" if risk_score < decision_threshold else "decline"
    credit_line = compute_credit_line(phase, risk_score)

    shap_output = explainer.shap_values(model_input)
    shap_vector = _extract_shap_vector(shap_output)

    contrib = pd.DataFrame(
        {
            "feature": model_input.columns,
            "value": model_input.iloc[0].to_numpy(dtype=float),
            "shap_contribution": np.asarray(shap_vector, dtype=float),
        }
    )
    contrib["abs_shap"] = contrib["shap_contribution"].abs()
    top5 = contrib.sort_values("abs_shap", ascending=False).head(5)

    top_drivers: list[TopDriver] = []
    for _, row in top5.iterrows():
        direction = "increased risk" if float(row["shap_contribution"]) >= 0 else "reduced risk"
        top_drivers.append(
            TopDriver(
                feature=get_feature_label(str(row["feature"])),
                value=float(row["value"]),
                shap_contribution=float(row["shap_contribution"]),
                direction=direction,
            )
        )

    explanation = _build_explanation_sentence(top_drivers)

    return PredictionResponse(
        risk_score=risk_score,
        decision=decision,
        cold_start_phase=phase,
        credit_line_MXN=credit_line,
        top_drivers=top_drivers,
        explanation=explanation,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = get_model() is not None
    return {"status": "healthy", "model_loaded": loaded}


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

    return {
        "title": "BorrowerInput",
        "fields": fields,
    }
