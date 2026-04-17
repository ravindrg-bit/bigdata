from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer


def load_table(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")
    return pd.read_parquet(path)


def compute_neighbor_aggregate(
    n_nodes: int,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    deg = np.bincount(src_idx, minlength=n_nodes) + np.bincount(dst_idx, minlength=n_nodes)
    value_sum = np.bincount(src_idx, weights=values[dst_idx], minlength=n_nodes) + np.bincount(
        dst_idx, weights=values[src_idx], minlength=n_nodes
    )
    out = np.full(n_nodes, float(values.mean()), dtype=float)
    has_neighbors = deg > 0
    out[has_neighbors] = value_sum[has_neighbors] / deg[has_neighbors]
    return out


def build_graph_features(
    borrowers: pd.DataFrame,
    edges: pd.DataFrame,
    labels: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    borrower_ids = borrowers["borrower_id"].to_numpy(dtype=np.int64)
    n = len(borrower_ids)

    id_to_idx = pd.Series(np.arange(n, dtype=np.int64), index=borrower_ids)
    src_idx = id_to_idx.loc[edges["src_id"].to_numpy()].to_numpy()
    dst_idx = id_to_idx.loc[edges["dst_id"].to_numpy()].to_numpy()

    edge_weight = edges["tie_strength"].to_numpy(dtype=float)
    degree = np.bincount(src_idx, minlength=n) + np.bincount(dst_idx, minlength=n)
    weighted_degree = np.bincount(src_idx, weights=edge_weight, minlength=n) + np.bincount(
        dst_idx, weights=edge_weight, minlength=n
    )

    label_map = labels.set_index("borrower_id")["default_flag"]
    defaults = label_map.loc[borrower_ids].to_numpy(dtype=float)

    nbr_default_rate_1hop = compute_neighbor_aggregate(n, src_idx, dst_idx, defaults)
    nbr_default_rate_2hop = compute_neighbor_aggregate(n, src_idx, dst_idx, nbr_default_rate_1hop)

    g = nx.Graph()
    g.add_nodes_from(borrower_ids.tolist())
    g.add_weighted_edges_from(
        zip(
            edges["src_id"].to_numpy(dtype=np.int64),
            edges["dst_id"].to_numpy(dtype=np.int64),
            edge_weight,
        )
    )

    if n > 60000:
        betweenness_k = 220
    elif n > 20000:
        betweenness_k = 450
    else:
        betweenness_k = min(700, n)

    pagerank = nx.pagerank(g, alpha=0.85, weight="weight", max_iter=100)
    betweenness = nx.betweenness_centrality(
        g,
        k=betweenness_k,
        normalized=True,
        seed=seed,
        weight="weight",
    )

    try:
        communities = nx.community.louvain_communities(g, weight="weight", seed=seed)
    except Exception:
        communities = nx.community.greedy_modularity_communities(g, weight="weight")

    community_index: dict[int, int] = {}
    for community_id, members in enumerate(communities):
        for node in members:
            community_index[int(node)] = community_id

    graph_features = pd.DataFrame(
        {
            "borrower_id": borrower_ids,
            "degree_centrality": degree / np.maximum(1, n - 1),
            "weighted_tie_strength": weighted_degree,
            "betweenness_centrality": np.array(
                [betweenness.get(int(bid), 0.0) for bid in borrower_ids], dtype=float
            ),
            "neighborhood_default_rate_1hop": nbr_default_rate_1hop,
            "neighborhood_default_rate_2hop": nbr_default_rate_2hop,
            "pagerank_score": np.array(
                [pagerank.get(int(bid), 0.0) for bid in borrower_ids], dtype=float
            ),
            "community_membership_flag": np.array(
                [community_index.get(int(bid), -1) for bid in borrower_ids], dtype=np.int64
            ),
        }
    )
    return graph_features


def build_features(data_dir: Path, output_path: Path, seed: int) -> pd.DataFrame:
    borrowers = load_table(data_dir, "borrowers")
    loans = load_table(data_dir, "loans")
    repayments = load_table(data_dir, "repayments")
    groups = load_table(data_dir, "groups")
    edges = load_table(data_dir, "edges")
    cdr = load_table(data_dir, "cdr")
    mobile = load_table(data_dir, "mobile_events")
    labels = load_table(data_dir, "labels")

    loan_agg = (
        loans.groupby("borrower_id", as_index=False)
        .agg(
            loan_count=("loan_id", "count"),
            avg_loan_amount_MXN=("amount_MXN", "mean"),
            max_loan_amount_MXN=("amount_MXN", "max"),
            cmr_credit_line_share=("CMR_credit_line_flag", "mean"),
            oxxo_cash_backed_share=("OXXO_cash_backed_flag", "mean"),
        )
        .reset_index(drop=True)
    )

    repayment_with_borrower = repayments.merge(
        loans[["loan_id", "borrower_id"]], on="loan_id", how="left"
    )
    repayment_agg = (
        repayment_with_borrower.groupby("borrower_id", as_index=False)
        .agg(
            repayment_latency_days=("repayment_latency_days", "mean"),
            on_time_repayment_share=("repayment_latency_days", lambda x: float((x <= 0).mean())),
        )
        .reset_index(drop=True)
    )

    behavioral = cdr[["borrower_id", "call_routine_score", "messaging_frequency", "weekly_call_cv"]].merge(
        mobile[
            [
                "borrower_id",
                "app_opens",
                "location_variance",
                "Falabella_app_session_flag",
                "routine_entropy",
                "codi_txn_regularity",
                "app_session_recency_days",
            ]
        ],
        on="borrower_id",
        how="left",
    )

    behavioral = behavioral.rename(
        columns={
            "weekly_call_cv": "call_volume_stability",
            "call_routine_score": "routine_score",
            "app_session_recency_days": "falabella_app_session_recency",
            "codi_txn_regularity": "CoDi_transaction_regularity",
        }
    )

    group_features = borrowers[["borrower_id", "group_id", "prior_CMR_usage", "store_visit_count", "CoDi_wallet_flag", "INE_verified_flag", "rural_flag", "indigenous_proxy", "age", "gender"]].merge(
        groups[["group_id", "cycle_number", "cohesion_score"]],
        on="group_id",
        how="left",
    )

    graph_features = build_graph_features(borrowers, edges, labels, seed=seed)

    feature_table = (
        borrowers[["borrower_id"]]
        .merge(group_features, on="borrower_id", how="left")
        .merge(loan_agg, on="borrower_id", how="left")
        .merge(repayment_agg, on="borrower_id", how="left")
        .merge(behavioral, on="borrower_id", how="left")
        .merge(graph_features, on="borrower_id", how="left")
        .merge(labels, on="borrower_id", how="left")
    )

    feature_table["sequential_lending_flag"] = (feature_table["loan_count"].fillna(0) >= 2).astype(int)
    feature_table["peer_default_contagion_score"] = feature_table[
        "neighborhood_default_rate_1hop"
    ]

    feature_table["gender_female_flag"] = (feature_table["gender"] == "F").astype(int)
    feature_table = feature_table.drop(columns=["gender"])

    numeric_columns = [col for col in feature_table.columns if col != "borrower_id"]
    imputer = SimpleImputer(strategy="median")
    feature_table[numeric_columns] = imputer.fit_transform(feature_table[numeric_columns])

    feature_table = feature_table.sort_values("borrower_id").reset_index(drop=True)

    if feature_table.isna().sum().sum() != 0:
        raise RuntimeError("Feature table still contains null values after imputation.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(feature_table, preserve_index=False),
        output_path,
        compression="snappy",
        use_dictionary=True,
    )
    return feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified feature table from synthetic risk data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing generated parquet files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Output feature table parquet path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for graph algorithms.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_features(args.data_dir, args.output, seed=args.seed)
    print(f"Features generated: {len(features):,} rows, {len(features.columns)} columns")
    print(f"Output path: {args.output}")


if __name__ == "__main__":
    main()
