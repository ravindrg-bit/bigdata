from __future__ import annotations

from typing import Literal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GraphVerificationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verification_method: Literal["linked_existing_nodes"] = Field(
        ...,
        description="Verification mode for manual graph claims",
    )
    linked_borrower_ids: list[int] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Existing borrower IDs with verified linkage evidence",
    )
    verifier_reference: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional verifier reference or ticket ID",
    )


class BorrowerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Identity and demographics
    age: int = Field(..., ge=18, le=72, description="Borrower age")
    gender: Literal["Female", "Male"] = Field(..., description="Borrower gender")
    rural_flag: Literal[0, 1] = Field(..., description="1 if rural, else 0")
    indigenous_proxy: Literal[0, 1] = Field(..., description="1 if indigenous proxy, else 0")
    INE_verified_flag: Literal[0, 1] = Field(..., description="1 if INE is verified, else 0")
    married_flag: Literal[0, 1] = Field(..., description="1 if married/in partnership, else 0")
    num_children: int = Field(..., ge=0, le=6, description="Number of children")

    # Financial profile
    monthly_income_MXN: float = Field(..., ge=1200, le=150000, description="Monthly income in MXN")
    rent_MXN: float = Field(..., ge=0, le=15000, description="Monthly rent in MXN")
    informal_loans_flag: Literal[0, 1] = Field(
        ..., description="1 if informal loans active, else 0"
    )
    formal_debt_flag: Literal[0, 1] = Field(..., description="1 if formal debt active, else 0")
    electricity_water_MXN: float = Field(..., ge=0, le=2000, description="Utilities spend in MXN")

    # Banco Falabella relationship
    store_visit_count: int = Field(..., ge=0, le=30, description="Store visits")
    prior_CMR_usage: int = Field(..., ge=0, le=14, description="Prior CMR usage count")
    CoDi_wallet_flag: Literal[0, 1] = Field(..., description="1 if CoDi wallet active, else 0")

    # Behavioral signals
    call_routine_score: float = Field(..., ge=0.0, le=1.0, description="Call routine score")
    messaging_frequency: float = Field(..., ge=0, le=200, description="Messaging frequency")
    call_volume_stability: float = Field(..., ge=0.0, le=1.0, description="Call volume stability")
    app_opens: int = Field(..., ge=0, le=100, description="App opens")
    location_variance: float = Field(..., ge=0.0, le=1.0, description="Location variance")

    # Social network
    peer_connections: int = Field(..., ge=0, le=20, description="Peer connection count")
    group_cohesion_score: float = Field(..., ge=0.0, le=1.0, description="Group cohesion score")
    neighborhood_default_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Neighborhood default rate"
    )
    graph_verification: Optional[GraphVerificationInput] = Field(
        default=None,
        description=(
            "Optional verified graph linkage payload. If omitted or invalid, manual scoring "
            "uses conservative graph defaults and zero embeddings."
        ),
    )


class TopDriver(BaseModel):
    feature: str
    value: Optional[float] = None
    shap_contribution: float
    direction: Literal["increased risk", "reduced risk"]


class PredictionResponse(BaseModel):
    raw_score: float
    risk_percentile: float
    risk_band: Literal["low", "medium", "high"]
    risk_score: float
    decision: Literal["approve", "approve_with_conditions", "decline"]
    cold_start_phase: Literal[1, 2, 3]
    credit_line_MXN: int
    top_drivers: list[TopDriver]
    explanation: str
    borrower_id: Optional[int] = None
    actual_outcome: Optional[str] = None
    model_correct: Optional[bool] = None


class BorrowerProfile(BaseModel):
    # Identity
    borrower_id: int
    age: int
    gender: str
    city: str
    rural_flag: int
    indigenous_proxy: int
    INE_verified_flag: int
    married_flag: int
    num_children: int
    CURP_hash: Optional[str] = None

    # Financial
    monthly_income_MXN: float
    rent_MXN: float
    mobile_phone_plan_MXN: float
    electricity_water_MXN: float
    informal_loans_MXN: float
    formal_debt_payments_MXN: float

    # Banco Falabella relationship
    store_visit_count: int
    prior_CMR_usage: Optional[int] = None
    CoDi_wallet_flag: int

    # Group lending
    group_id: int
    cohesion_score: float
    cycle_number: int

    # Loan history
    loan_count: int
    avg_loan_amount: float
    max_loan_amount: float
    total_loan_amount: float
    product_types: str
    CMR_credit_line_share: float
    OXXO_cash_backed_share: float

    # Repayment history
    mean_repayment_latency: float
    on_time_repayment_share: float
    worst_latency: float
    total_repayments: int

    # CDR behavioural signals
    call_volume: float
    call_routine_score: float
    messaging_frequency: float
    weekly_call_cv: float

    # Mobile / app signals
    app_opens: int
    location_variance: float
    Falabella_app_session_flag: int
    routine_entropy: float
    codi_txn_regularity: float
    app_session_recency_days: float

    # Social network
    peer_connection_count: int
    avg_tie_strength: float
    whatsapp_link_share: float
    codi_transfer_link_share: float

    # Graph-derived (pre-computed)
    degree_centrality: float
    betweenness_centrality: float
    neighborhood_default_rate_1hop: float
    neighborhood_default_rate_2hop: float
    pagerank_score: float
    community_id: int

    # Ground truth
    default_flag: int
    actual_outcome: str


class BorrowerSearchResult(BaseModel):
    borrower_id: int
    age: int
    gender: str
    city: str
    rural_flag: int
    loan_count: int
    default_flag: int


class BorrowerListResponse(BaseModel):
    total_borrowers: int
    results: list[BorrowerSearchResult]
