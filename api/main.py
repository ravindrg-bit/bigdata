from __future__ import annotations

import random
from typing import Any
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware

try:
    from .data_loader import BORROWER_ID_LIST, BORROWER_PROFILES, EMBEDDINGS_TABLE, FEATURES_TABLE
    from .feature_builder import (
        build_feature_vector,
        build_feature_vector_from_existing,
        compute_credit_line,
        determine_cold_start_phase,
        determine_cold_start_phase_existing,
    )
    from .model_loader import get_decision_threshold, get_explainer, get_feature_label, get_model
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
        build_feature_vector,
        build_feature_vector_from_existing,
        compute_credit_line,
        determine_cold_start_phase,
        determine_cold_start_phase_existing,
    )
    from model_loader import get_decision_threshold, get_explainer, get_feature_label, get_model
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
    ],
    allow_origin_regex=r"https://.*\.lovable\.app",
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

EXISTING_FEATURE_COLUMNS = [c for c in FEATURES_TABLE.columns if c != "borrower_id"]
EXISTING_EMBEDDING_COLUMNS = sorted([c for c in EMBEDDINGS_TABLE.columns if c.startswith("emb_")])


ENDPOINT_DESCRIPTIONS = [
    {"method": "GET", "path": "/health", "description": "Health check"},
    {"method": "GET", "path": "/stats", "description": "Dataset summary stats"},
    {"method": "GET", "path": "/schema", "description": "API schema and endpoint docs"},
    {"method": "GET", "path": "/borrower/random", "description": "Random borrower profile"},
    {"method": "GET", "path": "/borrower/search", "description": "Search by borrower ID prefix or city"},
    {"method": "GET", "path": "/borrower/{borrower_id}", "description": "Lookup specific borrower profile"},
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
    for col in model_features:
        if col not in x.columns:
            x[col] = 0.0
    return x[model_features].astype(float)


def _build_model_input(feature_df: pd.DataFrame, phase: int) -> pd.DataFrame:
    x = feature_df.drop(columns=["borrower_id", "default_flag"], errors="ignore").copy()
    if phase == 3:
        embedding_cols = [
            col for col in getattr(get_model(), "feature_names_in_", []) if str(col).startswith("emb_")
        ]
        for col in embedding_cols:
            if col not in x.columns:
                x[col] = 0.0
    return _align_to_model_features(x)


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


def _score_model_input(model_input: pd.DataFrame) -> tuple[float, list[TopDriver], str]:
    model = get_model()
    explainer = get_explainer()

    risk_score = float(model.predict_proba(model_input)[0, 1])

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
    return risk_score, top_drivers, explanation


@app.post("/predict/manual", response_model=PredictionResponse)
def predict_manual(payload: BorrowerInput) -> PredictionResponse:
    decision_threshold = get_decision_threshold()

    feature_df = build_feature_vector(payload)
    phase = determine_cold_start_phase(payload)
    model_input = _build_model_input(feature_df, phase)

    risk_score, top_drivers, explanation = _score_model_input(model_input)
    decision = "approve" if risk_score < decision_threshold else "decline"
    credit_line = compute_credit_line(phase, risk_score)

    return PredictionResponse(
        risk_score=risk_score,
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

    feature_vector, embedding_vector = build_feature_vector_from_existing(
        borrower_id=borrower_id,
        features_table=FEATURES_TABLE,
        embeddings_table=EMBEDDINGS_TABLE,
    )
    combined_values = np.concatenate([feature_vector, embedding_vector], axis=0)
    combined_columns = EXISTING_FEATURE_COLUMNS + EXISTING_EMBEDDING_COLUMNS
    existing_input = pd.DataFrame([combined_values], columns=combined_columns)
    model_input = _align_to_model_features(existing_input)

    risk_score, top_drivers, explanation = _score_model_input(model_input)
    decision = "approve" if risk_score < 0.5 else "decline"

    phase = determine_cold_start_phase_existing(row.to_dict())
    credit_line = compute_credit_line(phase, risk_score)

    default_flag = int(_to_native(row.get("default_flag")) or 0)
    actual_outcome = "defaulted" if default_flag == 1 else "repaid"
    model_correct = bool(
        (decision == "approve" and default_flag == 0)
        or (decision == "decline" and default_flag == 1)
    )

    return PredictionResponse(
        risk_score=risk_score,
        decision=decision,
        cold_start_phase=phase,
        credit_line_MXN=credit_line,
        top_drivers=top_drivers,
        explanation=explanation,
        borrower_id=borrower_id,
        actual_outcome=actual_outcome,
        model_correct=model_correct,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = get_model() is not None
    return {"status": "healthy", "model_loaded": loaded}


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
        raise HTTPException(status_code=404, detail=f"No borrowers available for default_status={default_status}")

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

    return {
        "title": "BorrowerInput",
        "fields": fields,
        "endpoints": ENDPOINT_DESCRIPTIONS,
    }
