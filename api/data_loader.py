from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


BORROWERS_PATH = RAW_DIR / "borrowers.parquet"
GROUPS_PATH = RAW_DIR / "groups.parquet"
LOANS_PATH = RAW_DIR / "loans.parquet"
REPAYMENTS_PATH = RAW_DIR / "repayments.parquet"
EDGES_PATH = RAW_DIR / "edges.parquet"
CDR_PATH = RAW_DIR / "cdr.parquet"
MOBILE_EVENTS_PATH = RAW_DIR / "mobile_events.parquet"
LABELS_PATH = RAW_DIR / "labels.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "gnn_embeddings.parquet"


GRAPH_FEATURE_COLUMNS = [
    "borrower_id",
    "degree_centrality",
    "weighted_tie_strength",
    "betweenness_centrality",
    "neighborhood_default_rate_1hop",
    "neighborhood_default_rate_2hop",
    "pagerank_score",
    "community_membership_flag",
]


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _optimize_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.columns:
        if str(out[col].dtype) == "Int64":
            out[col] = out[col].astype("Int32")
    return out


def _load_features_table() -> pd.DataFrame:
    table = pd.read_parquet(FEATURES_PATH)
    return _optimize_numeric_dtypes(table)


def _load_embeddings_table() -> pd.DataFrame:
    table = pd.read_parquet(EMBEDDINGS_PATH)
    return _optimize_numeric_dtypes(table)


def load_borrower_profiles() -> pd.DataFrame:
    borrowers = pd.read_parquet(BORROWERS_PATH)

    groups = pd.read_parquet(GROUPS_PATH)[["group_id", "cohesion_score", "cycle_number"]]
    borrowers = borrowers.merge(groups, on="group_id", how="left")

    labels = pd.read_parquet(LABELS_PATH)[["borrower_id", "default_flag"]]
    borrowers = borrowers.merge(labels, on="borrower_id", how="left")

    loans = pd.read_parquet(LOANS_PATH)
    loan_agg = (
        loans.groupby("borrower_id", as_index=False)
        .agg(
            loan_count=("loan_id", "count"),
            avg_loan_amount=("amount_MXN", "mean"),
            max_loan_amount=("amount_MXN", "max"),
            total_loan_amount=("amount_MXN", "sum"),
            product_types=(
                "product_type",
                lambda s: ",".join(sorted({str(v) for v in s.dropna().tolist()})),
            ),
            CMR_credit_line_share=("CMR_credit_line_flag", "mean"),
            OXXO_cash_backed_share=("OXXO_cash_backed_flag", "mean"),
        )
    )
    borrowers = borrowers.merge(loan_agg, on="borrower_id", how="left")

    repayments = pd.read_parquet(REPAYMENTS_PATH)
    repayment_base = repayments.merge(loans[["loan_id", "borrower_id"]], on="loan_id", how="left")
    repayment_agg = (
        repayment_base.groupby("borrower_id", as_index=False)
        .agg(
            mean_repayment_latency=("repayment_latency_days", "mean"),
            on_time_repayment_share=(
                "repayment_latency_days",
                lambda s: float((s <= 0).mean()) if len(s) else 0.0,
            ),
            worst_latency=("repayment_latency_days", "max"),
            total_repayments=("repayment_latency_days", "count"),
        )
    )
    borrowers = borrowers.merge(repayment_agg, on="borrower_id", how="left")

    cdr = pd.read_parquet(CDR_PATH)[
        [
            "borrower_id",
            "call_volume",
            "call_routine_score",
            "messaging_frequency",
            "weekly_call_cv",
        ]
    ]
    borrowers = borrowers.merge(cdr, on="borrower_id", how="left")

    mobile = pd.read_parquet(MOBILE_EVENTS_PATH)[
        [
            "borrower_id",
            "app_opens",
            "location_variance",
            "Falabella_app_session_flag",
            "routine_entropy",
            "codi_txn_regularity",
            "app_session_recency_days",
        ]
    ]
    borrowers = borrowers.merge(mobile, on="borrower_id", how="left")

    edges = pd.read_parquet(EDGES_PATH).reset_index(names="edge_id")
    src_side = edges[
        ["edge_id", "src_id", "tie_strength", "WhatsApp_metadata_proxy", "CoDi_transfer_link"]
    ].rename(columns={"src_id": "borrower_id"})
    dst_side = edges[
        ["edge_id", "dst_id", "tie_strength", "WhatsApp_metadata_proxy", "CoDi_transfer_link"]
    ].rename(columns={"dst_id": "borrower_id"})
    edge_long = pd.concat([src_side, dst_side], ignore_index=True)

    edge_agg = (
        edge_long.groupby("borrower_id", as_index=False)
        .agg(
            peer_connection_count=("edge_id", "nunique"),
            avg_tie_strength=("tie_strength", "mean"),
            whatsapp_link_share=("WhatsApp_metadata_proxy", "mean"),
            codi_transfer_link_share=("CoDi_transfer_link", "mean"),
        )
    )
    borrowers = borrowers.merge(edge_agg, on="borrower_id", how="left")

    graph_features = FEATURES_TABLE[GRAPH_FEATURE_COLUMNS].copy()
    borrowers = borrowers.merge(graph_features, on="borrower_id", how="left")

    borrowers["default_flag"] = borrowers["default_flag"].fillna(0)
    borrowers["product_types"] = borrowers["product_types"].fillna("")

    fill_zero_columns = [
        "cohesion_score",
        "cycle_number",
        "loan_count",
        "avg_loan_amount",
        "max_loan_amount",
        "total_loan_amount",
        "CMR_credit_line_share",
        "OXXO_cash_backed_share",
        "mean_repayment_latency",
        "on_time_repayment_share",
        "worst_latency",
        "total_repayments",
        "call_volume",
        "call_routine_score",
        "messaging_frequency",
        "weekly_call_cv",
        "app_opens",
        "location_variance",
        "Falabella_app_session_flag",
        "routine_entropy",
        "codi_txn_regularity",
        "app_session_recency_days",
        "peer_connection_count",
        "avg_tie_strength",
        "whatsapp_link_share",
        "codi_transfer_link_share",
        "degree_centrality",
        "weighted_tie_strength",
        "betweenness_centrality",
        "neighborhood_default_rate_1hop",
        "neighborhood_default_rate_2hop",
        "pagerank_score",
        "community_membership_flag",
    ]
    for col in fill_zero_columns:
        borrowers[col] = borrowers[col].fillna(0)

    borrowers = _optimize_numeric_dtypes(borrowers)

    mem_bytes = int(borrowers.memory_usage(deep=True).sum())
    print(
        "[data_loader] BORROWER_PROFILES shape="
        f"{borrowers.shape} rows_confirmed={len(borrowers) == 90000}"
    )
    print(f"[data_loader] BORROWER_PROFILES memory={_format_bytes(mem_bytes)}")
    print("[data_loader] BORROWER_PROFILES columns=")
    for col in borrowers.columns:
        print(f"  - {col}")

    return borrowers


FEATURES_TABLE = _load_features_table()
EMBEDDINGS_TABLE = _load_embeddings_table()
BORROWER_PROFILES = load_borrower_profiles()
BORROWER_ID_LIST = sorted(BORROWER_PROFILES["borrower_id"].astype(int).tolist())


_features_mem = int(FEATURES_TABLE.memory_usage(deep=True).sum())
_embeddings_mem = int(EMBEDDINGS_TABLE.memory_usage(deep=True).sum())
_profiles_mem = int(BORROWER_PROFILES.memory_usage(deep=True).sum())
_ids_mem = int(sys.getsizeof(BORROWER_ID_LIST) + sum(sys.getsizeof(i) for i in BORROWER_ID_LIST))
_total_mem = _features_mem + _embeddings_mem + _profiles_mem + _ids_mem

print(
    "[data_loader] startup_memory "
    f"profiles={_format_bytes(_profiles_mem)} "
    f"features={_format_bytes(_features_mem)} "
    f"embeddings={_format_bytes(_embeddings_mem)} "
    f"borrower_id_list={_format_bytes(_ids_mem)} "
    f"total={_format_bytes(_total_mem)}"
)