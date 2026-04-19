from __future__ import annotations

from collections import OrderedDict

import pandas as pd

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


def build_feature_vector(input_data: BorrowerInput) -> pd.DataFrame:
    """Build a 1-row DataFrame in exact features.parquet order.

    Notes:
    - Some loan-officer form fields are not present in the current 38-column
      training table schema. They are retained in request validation but are
      not directly emitted as columns because the parquet schema is authoritative.
    - Inference alignment to model.feature_names_in_ is handled in api/main.py.
    """

    d = input_data.model_dump()
    neighborhood_default_rate = float(d["neighborhood_default_rate"])
    peer_connections = int(d["peer_connections"])
    codi_wallet = int(d["CoDi_wallet_flag"])
    app_opens = int(d["app_opens"])

    informal_loans_mxn = 900.0 if int(d["informal_loans_flag"]) == 1 else 0.0
    formal_debt_payments_mxn = 1915.0 if int(d["formal_debt_flag"]) == 1 else 0.0
    mobile_phone_plan_mxn = 379.0

    # Derived auxiliary context values requested by spec; retained for traceability.
    _ = {
        "married_flag": int(d["married_flag"]),
        "num_children": int(d["num_children"]),
        "monthly_income_MXN": float(d["monthly_income_MXN"]),
        "rent_MXN": float(d["rent_MXN"]),
        "informal_loans_MXN": informal_loans_mxn,
        "formal_debt_payments_MXN": formal_debt_payments_mxn,
        "electricity_water_MXN": float(d["electricity_water_MXN"]),
        "mobile_phone_plan_MXN": mobile_phone_plan_mxn,
    }

    feature_values: OrderedDict[str, float | int] = OrderedDict(
        [
            ("borrower_id", -1),
            ("group_id", 0),
            ("prior_CMR_usage", int(d["prior_CMR_usage"])),
            ("store_visit_count", int(d["store_visit_count"])),
            ("CoDi_wallet_flag", codi_wallet),
            ("INE_verified_flag", int(d["INE_verified_flag"])),
            ("rural_flag", int(d["rural_flag"])),
            ("indigenous_proxy", int(d["indigenous_proxy"])),
            ("age", int(d["age"])),
            ("cycle_number", 2),
            ("cohesion_score", float(d["group_cohesion_score"])),
            ("loan_count", 1),
            ("avg_loan_amount_MXN", 5000.0),
            ("max_loan_amount_MXN", 5000.0),
            ("cmr_credit_line_share", 0.3),
            ("oxxo_cash_backed_share", 0.2),
            ("repayment_latency_days", 2.0),
            ("on_time_repayment_share", 0.85),
            ("routine_score", float(d["call_routine_score"])),
            ("messaging_frequency", float(d["messaging_frequency"])),
            ("call_volume_stability", float(d["call_volume_stability"])),
            ("app_opens", app_opens),
            ("location_variance", float(d["location_variance"])),
            ("Falabella_app_session_flag", 1 if app_opens > 0 else 0),
            ("routine_entropy", 0.5),
            ("CoDi_transaction_regularity", 0.5 if codi_wallet == 1 else 0.0),
            ("falabella_app_session_recency", 7 if app_opens > 0 else 30),
            ("degree_centrality", float(peer_connections) / 90000.0),
            ("weighted_tie_strength", 0.5),
            ("betweenness_centrality", 0.0001),
            ("neighborhood_default_rate_1hop", neighborhood_default_rate),
            ("neighborhood_default_rate_2hop", neighborhood_default_rate * 0.6),
            ("pagerank_score", 0.00005),
            ("community_membership_flag", 1 if peer_connections >= 3 else 0),
            ("default_flag", 0),
            ("sequential_lending_flag", 0),
            (
                "peer_default_contagion_score",
                neighborhood_default_rate * float(peer_connections) * 0.1,
            ),
            ("gender_female_flag", 1 if d["gender"] == "Female" else 0),
        ]
    )

    # Required by request: print full mapping before inference.
    print("[feature_builder] 38-feature mapping (features.parquet order):")
    for idx, col in enumerate(FEATURE_COLUMNS, start=1):
        print(f"  {idx:02d}. {col} = {feature_values[col]}")

    return pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)


def determine_cold_start_phase(input_data: BorrowerInput) -> int:
    if (
        input_data.store_visit_count < 3
        and input_data.CoDi_wallet_flag == 0
        and input_data.peer_connections == 0
    ):
        return 1
    if input_data.peer_connections == 0:
        return 2
    return 3


def compute_credit_line(phase: int, risk_score: float) -> int:
    phase_limits = {
        1: (500, 2000),
        2: (2000, 8000),
        3: (8000, 25000),
    }
    low, high = phase_limits[phase]
    return int(high - (high - low) * float(risk_score))
