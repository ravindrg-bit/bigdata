from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 42
    n_borrowers: int = 90000
    avg_group_size: int = 12
    target_default_rate: float = 0.18
    output_dir: Path = Path("data")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_intercept(logits: np.ndarray, target_rate: float) -> float:
    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        mean_rate = sigmoid(logits + mid).mean()
        if mean_rate < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def stable_curp_hash(ids: np.ndarray, seed: int) -> list[str]:
    return [
        hashlib.sha256(f"{seed}-{int(borrower_id)}".encode("utf-8")).hexdigest()[:18]
        for borrower_id in ids
    ]


def build_groups(cfg: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    n_groups = max(2500, cfg.n_borrowers // cfg.avg_group_size)
    group_ids = np.arange(1, n_groups + 1, dtype=np.int64)
    groups = pd.DataFrame(
        {
            "group_id": group_ids,
            "cycle_number": rng.integers(1, 9, size=n_groups, endpoint=False),
            "cohesion_score": np.clip(rng.beta(3.5, 2.0, size=n_groups), 0.05, 0.99),
        }
    )
    return groups


def build_borrowers(
    cfg: GenerationConfig, rng: np.random.Generator, groups: pd.DataFrame
) -> pd.DataFrame:
    n = cfg.n_borrowers
    borrower_ids = np.arange(1, n + 1, dtype=np.int64)

    group_probs = groups["cohesion_score"].to_numpy(dtype=float)
    group_probs = group_probs / group_probs.sum()
    group_assignment = rng.choice(groups["group_id"].to_numpy(), size=n, p=group_probs)

    age = np.clip(rng.normal(36, 10, size=n), 18, 72).round().astype(np.int64)
    gender = rng.choice(np.array(["F", "M"]), size=n, p=[0.53, 0.47])
    rural_flag = rng.binomial(1, 0.34, size=n).astype(np.int64)
    indigenous_proxy = (
        rng.random(size=n) < np.clip(0.08 + 0.18 * rural_flag, 0.02, 0.45)
    ).astype(np.int64)

    ine_prob = np.clip(0.92 - 0.13 * indigenous_proxy - 0.06 * rural_flag, 0.55, 0.99)
    ine_verified = rng.binomial(1, ine_prob, size=n).astype(np.int64)

    store_visit_count = (rng.poisson(3.8, size=n) + 1).astype(np.int64)

    prior_cmr_mask = rng.random(size=n) < 0.28
    prior_cmr_usage_raw = np.where(prior_cmr_mask, rng.integers(1, 15, size=n), np.nan)
    prior_cmr_usage = pd.Series(prior_cmr_usage_raw).round().astype("Int64")

    codi_prob = sigmoid(
        -0.7
        + 0.19 * store_visit_count
        - 0.03 * (age - 35)
        - 0.45 * rural_flag
        + 0.22 * prior_cmr_mask.astype(float)
        + 0.16 * ine_verified
    )
    codi_wallet_flag = rng.binomial(1, np.clip(codi_prob, 0.02, 0.98), size=n).astype(np.int64)

    borrowers = pd.DataFrame(
        {
            "borrower_id": borrower_ids,
            "age": age,
            "gender": gender,
            "rural_flag": rural_flag,
            "indigenous_proxy": indigenous_proxy,
            "CURP_hash": stable_curp_hash(borrower_ids, cfg.seed),
            "INE_verified_flag": ine_verified,
            "store_visit_count": store_visit_count,
            "prior_CMR_usage": prior_cmr_usage,
            "CoDi_wallet_flag": codi_wallet_flag,
            "group_id": group_assignment,
        }
    )
    return borrowers


def build_edges(
    borrowers: pd.DataFrame, groups: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    group_to_members = borrowers.groupby("group_id", sort=False)["borrower_id"].apply(np.array)
    group_to_cohesion = groups.set_index("group_id")["cohesion_score"].to_dict()

    edge_pairs: set[tuple[int, int]] = set()

    for group_id, members in group_to_members.items():
        size = len(members)
        if size < 2:
            continue
        cohesion = float(group_to_cohesion.get(group_id, 0.5))
        n_intra_edges = int(size * (1.8 + 2.8 * cohesion))
        if n_intra_edges <= 0:
            continue

        sampled_u = rng.choice(members, size=n_intra_edges, replace=True)
        sampled_v = rng.choice(members, size=n_intra_edges, replace=True)
        for u, v in zip(sampled_u, sampled_v):
            if u == v:
                continue
            a, b = (int(u), int(v)) if u < v else (int(v), int(u))
            edge_pairs.add((a, b))

    borrowers_arr = borrowers["borrower_id"].to_numpy()
    n_cross = int(len(borrowers_arr) * 1.1)
    cross_u = rng.choice(borrowers_arr, size=n_cross, replace=True)
    cross_v = rng.choice(borrowers_arr, size=n_cross, replace=True)
    for u, v in zip(cross_u, cross_v):
        if u == v:
            continue
        a, b = (int(u), int(v)) if u < v else (int(v), int(u))
        edge_pairs.add((a, b))

    edges_np = np.array(list(edge_pairs), dtype=np.int64)
    if edges_np.size == 0:
        raise RuntimeError("No edges were generated. Increase borrower count or edge density.")

    src = edges_np[:, 0]
    dst = edges_np[:, 1]

    borrower_group = borrowers.set_index("borrower_id")["group_id"].to_dict()
    same_group = np.array([borrower_group[int(s)] == borrower_group[int(d)] for s, d in zip(src, dst)])

    tie_strength = np.clip(rng.beta(2.2, 1.7, size=len(src)) + 0.15 * same_group, 0.05, 1.0)
    whatsapp_prob = np.clip(0.18 + 0.65 * tie_strength, 0.02, 0.98)
    whatsapp_metadata_proxy = rng.binomial(1, whatsapp_prob, size=len(src)).astype(np.int64)

    codi_wallet = borrowers.set_index("borrower_id")["CoDi_wallet_flag"].to_dict()
    codi_link_prob = np.array(
        [
            0.08
            + 0.52 * (codi_wallet[int(s)] == 1 and codi_wallet[int(d)] == 1)
            + 0.14 * same
            for s, d, same in zip(src, dst, same_group)
        ],
        dtype=float,
    )
    codi_transfer_link = rng.binomial(1, np.clip(codi_link_prob, 0.01, 0.95), size=len(src)).astype(
        np.int64
    )

    edges = pd.DataFrame(
        {
            "src_id": src,
            "dst_id": dst,
            "tie_strength": tie_strength.round(4),
            "WhatsApp_metadata_proxy": whatsapp_metadata_proxy,
            "CoDi_transfer_link": codi_transfer_link,
        }
    )
    return edges


def build_cdr(borrowers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(borrowers)
    codi = borrowers["CoDi_wallet_flag"].to_numpy()
    rural = borrowers["rural_flag"].to_numpy()
    ine = borrowers["INE_verified_flag"].to_numpy()
    prior_cmr = borrowers["prior_CMR_usage"].notna().astype(int).to_numpy()

    call_volume = np.clip(rng.poisson(72 + 10 * codi - 8 * rural, size=n), 8, None)
    call_routine_score = np.clip(rng.normal(0.56 + 0.11 * ine, 0.16, size=n), 0.02, 0.99)
    messaging_frequency = np.clip(
        rng.poisson(38 + 12 * codi + 6 * prior_cmr - 6 * rural, size=n), 1, None
    )
    weekly_call_cv = np.clip(rng.normal(0.55 - 0.25 * call_routine_score, 0.12, size=n), 0.05, 1.5)

    cdr = pd.DataFrame(
        {
            "borrower_id": borrowers["borrower_id"].to_numpy(),
            "call_volume": call_volume.astype(np.int64),
            "call_routine_score": call_routine_score.round(4),
            "messaging_frequency": messaging_frequency.astype(np.int64),
            "weekly_call_cv": weekly_call_cv.round(4),
        }
    )
    return cdr


def build_mobile_events(borrowers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(borrowers)
    codi = borrowers["CoDi_wallet_flag"].to_numpy()
    rural = borrowers["rural_flag"].to_numpy()
    prior_cmr = borrowers["prior_CMR_usage"].notna().astype(int).to_numpy()

    app_opens = np.clip(rng.poisson(26 + 10 * codi + 5 * prior_cmr, size=n), 0, None)
    location_variance = np.clip(rng.normal(0.45 + 0.18 * rural, 0.15, size=n), 0.01, 1.5)

    app_session_prob = sigmoid(-1.0 + 0.065 * app_opens + 0.75 * codi)
    falabella_app_session_flag = rng.binomial(
        1, np.clip(app_session_prob, 0.02, 0.99), size=n
    ).astype(np.int64)

    routine_entropy = np.clip(
        rng.normal(0.64 - 0.26 * falabella_app_session_flag, 0.13, size=n), 0.05, 1.5
    )
    codi_txn_regularity = np.clip(rng.normal(0.5 + 0.25 * codi, 0.17, size=n), 0.02, 0.99)
    app_session_recency_days = np.clip(
        rng.normal(9 - 4.2 * falabella_app_session_flag + 2.8 * rural, 5.0, size=n), 0, 60
    ).round().astype(np.int64)

    mobile_events = pd.DataFrame(
        {
            "borrower_id": borrowers["borrower_id"].to_numpy(),
            "app_opens": app_opens.astype(np.int64),
            "location_variance": location_variance.round(4),
            "Falabella_app_session_flag": falabella_app_session_flag,
            "routine_entropy": routine_entropy.round(4),
            "codi_txn_regularity": codi_txn_regularity.round(4),
            "app_session_recency_days": app_session_recency_days,
        }
    )
    return mobile_events


def build_labels(
    cfg: GenerationConfig,
    borrowers: pd.DataFrame,
    edges: pd.DataFrame,
    cdr: pd.DataFrame,
    mobile_events: pd.DataFrame,
    groups: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(borrowers)
    age = borrowers["age"].to_numpy()
    rural = borrowers["rural_flag"].to_numpy()
    indigenous = borrowers["indigenous_proxy"].to_numpy()
    ine = borrowers["INE_verified_flag"].to_numpy()
    store_visits = borrowers["store_visit_count"].to_numpy()
    codi = borrowers["CoDi_wallet_flag"].to_numpy()
    prior_cmr = borrowers["prior_CMR_usage"].notna().astype(int).to_numpy()
    group_ids = borrowers["group_id"].to_numpy()

    cohesion_by_group = groups.set_index("group_id")["cohesion_score"]
    cohesion = cohesion_by_group.loc[group_ids].to_numpy()

    call_routine = cdr["call_routine_score"].to_numpy()
    location_var = mobile_events["location_variance"].to_numpy()

    latent_risk = (
        0.22 * (age < 24)
        + 0.16 * (age > 58)
        + 0.3 * rural
        + 0.35 * indigenous
        - 0.33 * ine
        - 0.06 * np.log1p(store_visits)
        - 0.24 * codi
        - 0.12 * prior_cmr
        + 0.18 * location_var
        + 0.13 * (1.0 - call_routine)
        + rng.normal(0, 0.25, size=n)
    )

    src_idx = edges["src_id"].to_numpy(dtype=np.int64) - 1
    dst_idx = edges["dst_id"].to_numpy(dtype=np.int64) - 1

    deg = np.bincount(src_idx, minlength=n) + np.bincount(dst_idx, minlength=n)
    nbr_sum = np.bincount(src_idx, weights=latent_risk[dst_idx], minlength=n) + np.bincount(
        dst_idx, weights=latent_risk[src_idx], minlength=n
    )

    neighborhood_risk = np.full(n, latent_risk.mean(), dtype=float)
    mask = deg > 0
    neighborhood_risk[mask] = nbr_sum[mask] / deg[mask]

    logits = (
        1.12 * latent_risk
        + 1.05 * neighborhood_risk
        - 0.68 * cohesion
        + 0.09 * (1.0 - call_routine)
        + 0.06 * location_var
    )
    intercept = calibrate_intercept(logits, cfg.target_default_rate)
    default_prob = sigmoid(logits + intercept)
    default_flag = rng.binomial(1, np.clip(default_prob, 0.001, 0.999), size=n).astype(np.int64)

    labels = pd.DataFrame(
        {
            "borrower_id": borrowers["borrower_id"].to_numpy(dtype=np.int64),
            "default_flag": default_flag,
        }
    )
    return labels


def build_loans_and_repayments(
    borrowers: pd.DataFrame,
    labels: pd.DataFrame,
    groups: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    borrower_ids = borrowers["borrower_id"].to_numpy(dtype=np.int64)
    n = len(borrower_ids)

    loan_count = 1 + rng.binomial(1, 0.36, size=n) + rng.binomial(1, 0.09, size=n)
    borrower_rep = np.repeat(borrower_ids, loan_count)

    loan_id = np.arange(1, len(borrower_rep) + 1, dtype=np.int64)

    default_lookup = labels.set_index("borrower_id")["default_flag"]
    borrower_default = default_lookup.loc[borrower_rep].to_numpy(dtype=np.int64)

    borrower_store = borrowers.set_index("borrower_id")["store_visit_count"].loc[borrower_rep].to_numpy()
    borrower_prior = (
        borrowers.set_index("borrower_id")["prior_CMR_usage"].loc[borrower_rep].notna().astype(int).to_numpy()
    )
    borrower_codi = borrowers.set_index("borrower_id")["CoDi_wallet_flag"].loc[borrower_rep].to_numpy()
    borrower_rural = borrowers.set_index("borrower_id")["rural_flag"].loc[borrower_rep].to_numpy()

    group_cohesion = groups.set_index("group_id")["cohesion_score"]
    borrower_group = borrowers.set_index("borrower_id")["group_id"].loc[borrower_rep].to_numpy()
    cohesion_rep = group_cohesion.loc[borrower_group].to_numpy()

    product_type = rng.choice(
        np.array(["micro_loan", "working_capital", "appliance_finance", "cash_advance"]),
        size=len(loan_id),
        p=[0.44, 0.26, 0.2, 0.1],
    )

    base_amount = np.exp(np.log(2300 + 250 * borrower_store) + rng.normal(0, 0.52, size=len(loan_id)))
    amount_mxn = np.clip(base_amount * (1 + 0.2 * borrower_default), 500, 25000).round(2)

    cmr_prob = np.clip(0.16 + 0.34 * borrower_prior + 0.08 * borrower_codi, 0.05, 0.95)
    oxxo_prob = np.clip(0.22 + 0.18 * borrower_rural + 0.06 * (product_type == "cash_advance"), 0.05, 0.95)

    cmr_credit_line_flag = rng.binomial(1, cmr_prob, size=len(loan_id)).astype(np.int64)
    oxxo_cash_backed_flag = rng.binomial(1, oxxo_prob, size=len(loan_id)).astype(np.int64)

    origination_date = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 366, size=len(loan_id)), unit="D"
    )
    term_days = rng.choice(np.array([30, 45, 60, 90]), size=len(loan_id), p=[0.28, 0.3, 0.28, 0.14])
    due_date = origination_date + pd.to_timedelta(term_days, unit="D")

    latency = np.clip(
        np.round(
            rng.normal(
                loc=1.5
                + 6.0 * borrower_default
                + 1.3 * (1 - borrower_codi)
                - 1.6 * cohesion_rep,
                scale=4.7,
                size=len(loan_id),
            )
        ),
        -10,
        95,
    ).astype(np.int64)

    paid_date = due_date + pd.to_timedelta(latency, unit="D")

    repay_multiplier = np.where(
        borrower_default == 1,
        rng.uniform(0.55, 1.02, size=len(loan_id)),
        rng.uniform(0.98, 1.05, size=len(loan_id)),
    )
    repayment_amount = (amount_mxn * repay_multiplier).round(2)

    loans = pd.DataFrame(
        {
            "loan_id": loan_id,
            "borrower_id": borrower_rep,
            "amount_MXN": amount_mxn,
            "product_type": product_type,
            "CMR_credit_line_flag": cmr_credit_line_flag,
            "OXXO_cash_backed_flag": oxxo_cash_backed_flag,
        }
    )

    repayments = pd.DataFrame(
        {
            "loan_id": loan_id,
            "due_date": due_date,
            "paid_date": paid_date,
            "amount": repayment_amount,
            "repayment_latency_days": latency,
        }
    )

    return loans, repayments


def write_parquet(df: pd.DataFrame, output_path: Path, sort_columns: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stable_df = df.sort_values(sort_columns).reset_index(drop=True)
    table = pa.Table.from_pandas(stable_df, preserve_index=False)
    pq.write_table(
        table,
        output_path,
        compression="snappy",
        use_dictionary=True,
        version="2.6",
    )


def generate_dataset(cfg: GenerationConfig) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)

    groups = build_groups(cfg, rng)
    borrowers = build_borrowers(cfg, rng, groups)
    edges = build_edges(borrowers, groups, rng)
    cdr = build_cdr(borrowers, rng)
    mobile_events = build_mobile_events(borrowers, rng)
    labels = build_labels(cfg, borrowers, edges, cdr, mobile_events, groups, rng)
    loans, repayments = build_loans_and_repayments(borrowers, labels, groups, rng)

    return {
        "borrowers": borrowers,
        "loans": loans,
        "repayments": repayments,
        "groups": groups,
        "edges": edges,
        "cdr": cdr,
        "mobile_events": mobile_events,
        "labels": labels,
    }


def save_dataset(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sort_map = {
        "borrowers": ["borrower_id"],
        "loans": ["loan_id"],
        "repayments": ["loan_id"],
        "groups": ["group_id"],
        "edges": ["src_id", "dst_id"],
        "cdr": ["borrower_id"],
        "mobile_events": ["borrower_id"],
        "labels": ["borrower_id"],
    }

    for name, table in tables.items():
        write_parquet(table, output_dir / f"{name}.parquet", sort_map[name])


def summarize_outputs(output_dir: Path, labels: pd.DataFrame) -> None:
    total_bytes = 0
    print("Generated files:")
    for parquet_path in sorted(output_dir.glob("*.parquet")):
        size_mb = parquet_path.stat().st_size / (1024 * 1024)
        total_bytes += parquet_path.stat().st_size
        print(f"  - {parquet_path.name}: {size_mb:.2f} MB")

    total_mb = total_bytes / (1024 * 1024)
    default_rate = float(labels["default_flag"].mean())
    print(f"\nTotal parquet size: {total_mb:.2f} MB")
    print(f"Observed default rate: {default_rate:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Falabella-style risk data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--n-borrowers",
        type=int,
        default=90000,
        help="Number of borrowers to generate.",
    )
    parser.add_argument(
        "--target-default-rate",
        type=float,
        default=0.18,
        help="Target portfolio default rate.",
    )
    parser.add_argument(
        "--avg-group-size",
        type=int,
        default=12,
        help="Average borrower group size.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where parquet files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GenerationConfig(
        seed=args.seed,
        n_borrowers=args.n_borrowers,
        avg_group_size=args.avg_group_size,
        target_default_rate=args.target_default_rate,
        output_dir=args.output_dir,
    )

    tables = generate_dataset(cfg)
    save_dataset(tables, cfg.output_dir)
    summarize_outputs(cfg.output_dir, tables["labels"])


if __name__ == "__main__":
    main()
