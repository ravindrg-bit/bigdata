# Data Documentation — Falabella Risk Engine

---

## 1. Data Engineering Pipeline: Bronze → Silver → Gold

### Bronze Layer — Raw Synthetic Generation
The bronze layer contains the 8 raw parquet files as generated directly by `src/falabella_risk/data/generate_data.py`. No transformations have been applied. Data reflects synthetic construction calibrated to official Mexican demographic and financial statistics.

| File | Rows | Description |
|---|---|---|
| `borrowers.parquet` | 90,000 | One record per borrower. Identity, demographics, financial profile, behavioural signals |
| `groups.parquet` | 7,500 | Peer solidarity lending groups with cohesion scores |
| `loans.parquet` | 130,580 | Individual loan records (1–3 loans per borrower) |
| `repayments.parquet` | 130,580 | Repayment event per loan — dates, amounts, latency |
| `edges.parquet` | 334,955 | Social graph edges between borrower pairs |
| `cdr.parquet` | 90,000 | Mobile call-detail record proxies per borrower |
| `mobile_events.parquet` | 90,000 | App usage and location behaviour per borrower |
| `labels.parquet` | 90,000 | Ground-truth default flag (18% default rate) |

**Transformations at this layer:** None. Data is immutable once generated. Seed is fixed at 42 for reproducibility.

---

### Silver Layer — Feature Engineering
The silver layer joins and transforms all 8 bronze tables into a single flat feature table. Implemented in `src/falabella_risk/features/feature_engineering.py`.

**Steps performed:**

1. **Join borrower profile + group attributes** — selects 9 columns from `borrowers.parquet` (`prior_CMR_usage`, `store_visit_count`, `CoDi_wallet_flag`, `INE_verified_flag`, `rural_flag`, `indigenous_proxy`, `age`, `gender`, `group_id`) and joins `groups.parquet` to add `cycle_number` and `cohesion_score`. Note: the 9 additional borrower fields added for demographic richness (`monthly_income_MXN`, `rent_MXN`, `mobile_phone_plan_MXN`, `electricity_water_MXN`, `informal_loans_MXN`, `formal_debt_payments_MXN`, `married_flag`, `num_children`, `city`) reside in `borrowers.parquet` but are not yet selected by the feature pipeline — they are candidates for the next model iteration.
2. **Aggregate loan history** — from `loans.parquet`: loan count, average and maximum loan amount, CMR credit line share, OXXO cash-backed share
3. **Aggregate repayment behaviour** — from `repayments.parquet`: mean repayment latency, on-time repayment share
4. **Join CDR signals** — from `cdr.parquet`: `call_routine_score` (→ renamed `routine_score`), `messaging_frequency`, `weekly_call_cv` (→ renamed `call_volume_stability`). Note: `call_volume` is in the raw CDR file but is not selected for features.
5. **Join mobile events** — from `mobile_events.parquet`: `app_opens`, `location_variance`, `Falabella_app_session_flag`, `routine_entropy`, `codi_txn_regularity` (→ renamed `CoDi_transaction_regularity`), `app_session_recency_days` (→ renamed `falabella_app_session_recency`)
6. **Compute graph features** — from `edges.parquet` using NetworkX:
   - `degree_centrality` — how connected a borrower is in the peer network
   - `weighted_tie_strength` — average strength of social connections
   - `betweenness_centrality` — how often a borrower sits between others (broker position)
   - `neighborhood_default_rate_1hop` — default rate of direct peers
   - `neighborhood_default_rate_2hop` — default rate of peers-of-peers
   - `pagerank_score` — influence/trust score in the network
   - `community_membership_flag` — belonging to a dense sub-community
7. **Engineer derived flags** — `sequential_lending_flag`, `peer_default_contagion_score`, `gender_female_flag`
8. **Impute missing values** — median imputation for any null features
9. **Attach label** — join `default_flag` from `labels.parquet`

**Output:** `data/processed/features.parquet` — one row per borrower, 38 features, ready for model training.

---

### Gold Layer — Model-Ready Artifacts
The gold layer contains trained model binaries and GNN embeddings, produced by the training pipeline.

| Artifact | File | How produced |
|---|---|---|
| XGBoost baseline | `models/baseline_xgb.pkl` | Trained on `features.parquet` tabular features |
| GraphSAGE embeddings | `data/processed/gnn_embeddings.parquet` | Node embeddings from `edges.parquet` + node features |
| GraphSAGE model | `models/graphsage.pt` | Trained on graph + node features |
| Hybrid ensemble | `models/hybrid_ensemble.pkl` | XGBClassifier trained on concatenated tabular features + GraphSAGE embeddings |
| Federated model | `models/federated_model.pt` | FedAvg across 4 city nodes: CDMX, Monterrey, Guadalajara, Mérida |

**Downstream consumers:** Streamlit app (`src/falabella_risk/app/main.py`), cold-start scorer (`src/falabella_risk/inference/cold_start.py`), fairness audit (`src/falabella_risk/evaluation/fairness_audit.py`)

---

## 2. Exploratory Data Analysis — `borrowers.parquet`

**Dataset:** 90,000 borrowers × 20 columns  
**Target audience:** Thin-file micro-credit applicants in Mexico  
**Seed:** 42 (fully reproducible)

---

### 2.1 Gender Distribution

| Gender | Count | Share |
|---|---|---|
| Female | 63,135 | **70.2%** |
| Male | 26,865 | 29.8% |

> Calibrated to the Mexican micro-credit market profile. Compartamos Banco (largest microfinance institution in Latin America) reports ~70% female borrower composition. The model card explicitly flags `gender_female_flag` as a fairness-sensitive attribute.
>
> **Source:** Compartamos Banco Annual Report 2022; ProMujer Mexico programme data.

---

### 2.2 Age Distribution

| Band | Share | Count |
|---|---|---|
| 18–24 | 18.1% | 16,268 |
| 25–34 | 32.1% | 28,880 |
| **35–44** | **27.9%** | **25,074** |
| 45–54 | 14.9% | 13,435 |
| 55–64 | 6.0% | 5,444 |
| 65–72 | 1.0% | 899 |

**Mean age:** 35.9 years &nbsp;|&nbsp; **Median:** 34.0 years

> Distribution is intentionally skewed younger than the national adult population (INEGI national mean ~42 years). Thin-file micro-credit borrowers peak in the 25–44 working-age band. Older adults (55+) are limited to 7% as they typically have established credit histories and are not thin-file applicants.
>
> **Source:** INEGI Censo de Población y Vivienda 2020; Compartamos Banco / Gentera Mexico borrower profile data.

---

### 2.3 Rural / Urban Split

| Category | Count | Share |
|---|---|---|
| Urban | 71,293 | **79.2%** |
| Rural | 18,707 | 20.8% |

> Matches the INEGI 2020 Census urban/rural definition (localities < 2,500 inhabitants = rural). Rural borrowers are a fairness-sensitive group in the model — they show a higher default rate (24.3% vs 16.3% urban) driven by lower income, lower INE verification, and higher informal credit dependence.
>
> **Source:** INEGI CPV 2020 — Cuentame de México, Población Rural y Urbana. https://cuentame.inegi.org.mx/descubre/poblacion/rural_urbana/

---

### 2.4 City Distribution (Top 10)

| City | Count | Share |
|---|---|---|
| Other (urban) | 21,207 | 23.6% |
| Rural | 18,707 | 20.8% |
| Ciudad de México | 17,479 | 19.4% |
| Mérida | 7,377 | 8.2% ¹ |
| Monterrey | 4,400 | 4.9% |
| Guadalajara | 4,238 | 4.7% |
| Puebla | 2,582 | 2.9% |
| Toluca | 1,839 | 2.0% |
| Tijuana | 1,643 | 1.8% |
| León | 1,538 | 1.7% |

> ¹ Mérida is intentionally oversampled (population-proportional share would be ~1.3%) to ensure the Yucatán federated training node has sufficient records (~7,377 borrowers). The model card names Yucatán as one of 4 FedAvg clients.
>
> **Source:** INEGI CPV 2020 — Sistema Urbano Nacional, top ZMVMs by population.

---

### 2.5 Indigenous Population

| | Share |
|---|---|
| Non-indigenous | 82.1% |
| Indigenous proxy = 1 | **17.9%** |

> Calibrated to the INEGI 2020 self-identification figure. Urban indigenous baseline: 14%; rural: 33% — consistent with INEGI's spatial distribution showing indigenous populations are disproportionately rural.
>
> **Source:** INEGI CPV 2020 — Pueblos Indígenas press release 2022. 23.2M self-identified indigenous / 126M total = 18.4%. https://www.inegi.org.mx/contenidos/saladeprensa/aproposito/2022/EAP_PueblosInd22.pdf

---

### 2.6 Identity Verification (INE)

| | Share |
|---|---|
| INE verified | **90.5%** |
| Not verified | 9.5% |

> Mexico's INE (Instituto Nacional Electoral) electoral credential covers ~90% of the adult population. Unverified borrowers show a materially higher default rate (27.8% vs 16.9% verified), making this a strong credit signal for thin-file applicants.
>
> **Source:** INE Padrón Electoral 2022 — https://ine.mx/credencial/

---

### 2.7 Marital Status

| | Share |
|---|---|
| In partnership (married / unión libre) | **50.3%** |
| Single / separated / widowed | 49.7% |

> INEGI CPV 2020 reports 38% casado + 20% unión libre = 58% in partnership nationally for adults 15+. The dataset mean of 50% reflects a younger borrower population (18–44) where partnership rates are lower than the national adult average that includes older cohorts.
>
> **Source:** INEGI CPV 2020 — Nupcialidad. https://www.inegi.org.mx/temas/nupcialidad/

---

### 2.8 Number of Children

| Children | Count | Share |
|---|---|---|
| 0 | 54,218 | 60.2% |
| 1 | 18,926 | 21.0% |
| 2 | 9,641 | 10.7% |
| 3 | 4,377 | 4.9% |
| 4+ | 2,838 | 3.2% |

**Mean:** 0.71 children per borrower

> Younger borrower skew explains the high proportion with no children. Mean rises to ~1.4 for borrowers aged 35–54.
>
> **Source:** INEGI CPV 2020; CONAPO Proyecciones de Población 2020 — TFR 1.6 nationally; rural TFR ~2.1.

---

### 2.9 Monthly Income (MXN)

| Metric | All | Urban | Rural |
|---|---|---|---|
| Minimum | 1,200 | — | — |
| 25th percentile | 4,621 | — | — |
| **Median** | **7,446** | **8,326** | **4,654** |
| Mean | 9,406 | — | — |
| 75th percentile | 11,890 | — | — |
| Maximum | 144,253 | — | — |

> Log-normal distribution calibrated to INEGI ENIGH 2022. Urban individual median ~8,000 MXN/month; rural ~4,500 MXN/month. Gender wage gap of ~14% applied (women earn ~86% of men).
>
> **Source:** INEGI ENIGH 2022 — Comunicado de Prensa Núm. 420/23. https://www.inegi.org.mx/contenidos/programas/enigh/nc/2022/

---

### 2.10 Monthly Fixed Liabilities

| Liability | Coverage | Median (payers) |
|---|---|---|
| Electricity + Water | **90.3%** | 541 MXN |
| Rent | 30.7% | 5,144 MXN |
| Mobile phone plan (postpaid) | 30.7% | 379 MXN |
| Informal loans (tandas / prestamistas) | 22.1% | 900 MXN |
| Formal debt payments | 13.8% | 1,915 MXN |

> Formal debt is intentionally low (13.3% vs national 32.7%) reflecting the thin-file premise — borrowers with extensive formal credit are excluded from the target population.
>
> **Sources:**
> - Electricity/water: INEGI CPV 2020; CFE residential tariff data 2022
> - Rent: INEGI ENIGH 2022 (30% of households rent)
> - Mobile postpaid: IFT (Instituto Federal de Telecomunicaciones) 2022 — ~30% of lines are postpaid
> - Informal loans: ENIF 2021 — 23% of adults used informal credit
> - Formal debt: ENIF 2021 (INEGI/CNBV) — 32.7% nationally, adjusted for thin-file population

---

### 2.11 CoDi Digital Wallet

| | Share |
|---|---|
| Has CoDi wallet | **13.0%** |
| No CoDi wallet | 87.0% |

> CoDi (Cobro Digital) is Banxico's mobile payment system. National adoption peaked at ~10.7M accounts (~12% of adults) in September 2021. Higher adoption in urban/higher-income borrowers; a strong positive credit signal (CoDi=1 default rate: 14.5% vs CoDi=0: 18.4%).
>
> **Source:** Banxico — CoDi system statistics, September 2021. https://www.banxico.org.mx/sistemas-de-pago/codi-cobro-digital-banco-me.html

---

### 2.12 Thin-File Signal

| | Count | Share |
|---|---|---|
| No prior CMR usage (thin-file) | 65,018 | **72.2%** |
| Has prior CMR usage | 24,982 | 27.8% |

> The defining characteristic of the target population. 72.2% of borrowers have no prior CMR (Falabella store credit card) history — they are genuinely thin-file and unscored by traditional models. In the raw data, thin-file status is encoded as a null value in `prior_CMR_usage`; borrowers with history have a count of 1–14 prior uses. The feature pipeline median-imputes nulls, so thin-file borrowers are routed through the Phase 1 rule-based cold-start scorer rather than the full model.

---

## 3. Understanding Each Parquet File and How They Work Together

### The 8 Files and Their Purpose

---

**`borrowers.parquet`** — *Master borrower record*  
The central entity of the entire dataset. Contains every attribute observable about a borrower at application time: identity, demographics, financial profile, and behavioural signals. Every other file except `groups.parquet` links back to this file via `borrower_id`. For a loan officer, this is the application form. For the model, this provides the base feature set.

---

**`groups.parquet`** — *Solidarity lending group registry*  
Defines the peer groups that borrowers belong to — analogous to Grameen Bank solidarity circles. Each group has a `cohesion_score` (how tightly bonded the group is) and `cycle_number` (how many lending cycles the group has completed). High-cohesion groups create social accountability pressure that reduces default risk. Borrowers are assigned to groups via `borrowers.group_id`. The GraphSAGE GNN uses group membership to seed the social graph.

---

**`loans.parquet`** — *Individual loan records*  
One row per loan. Borrowers can hold 1–3 loans simultaneously (130,580 loans across 90,000 borrowers). Each loan has an amount, product type (`micro_loan`, `working_capital`, `appliance_finance`, `cash_advance`), and payment channel flags. Loan amount is positively correlated with store visit count (correlation: 0.23) — more engaged customers receive larger credit lines. Links to repayments via `loan_id`.

---

**`repayments.parquet`** — *Repayment event per loan*  
Matched 1:1 with loans. Records `due_date`, `paid_date`, `amount` paid, and `repayment_latency_days`. Negative latency means paid early; positive means paid late. Defaulters average 7.6 days late vs 1.5 days for non-defaulters. This file is the primary source of repayment behaviour features in the silver layer: `on_time_repayment_share` and mean latency.

---

**`edges.parquet`** — *Social network graph*  
Defines the peer relationship graph: 334,955 directed edges between borrower pairs. Each edge carries `tie_strength` (0.05–1.0), a `WhatsApp_metadata_proxy` flag (61.3% of edges have a communication link), and a `CoDi_transfer_link` flag (18.7% of edges have a digital payment link). This file feeds the GraphSAGE GNN directly — each borrower becomes a node, each edge a connection. Graph features derived from this file (neighborhood default rate, degree centrality, PageRank) are among the strongest predictors in the hybrid model.

---

**`cdr.parquet`** — *Mobile call-detail record proxies*  
One row per borrower. Contains four behavioural regularity signals: `call_volume` (mean 71.6 calls/period), `call_routine_score` (how predictable the call pattern is, mean 0.66), `messaging_frequency`, and `weekly_call_cv` (coefficient of variation — lower = more stable). For thin-file borrowers without formal credit history, these signals serve as an alternative creditworthiness proxy. Higher call routine scores correlate with lower default rates.

---

**`mobile_events.parquet`** — *Smartphone and app usage behaviour*  
One row per borrower. Six signals: `app_opens`, `location_variance` (higher in rural), `Falabella_app_session_flag` (71% have had a session), `routine_entropy` (predictability of daily routine), `codi_txn_regularity` (renamed to `CoDi_transaction_regularity` in features.parquet), and `app_session_recency_days` (renamed to `falabella_app_session_recency` in features.parquet). These signals are richer than CDR — the Falabella app session flag alone indicates commercial intent. Works in tandem with CDR to build a digital behavioural profile for thin-file scoring.

---

**`labels.parquet`** — *Ground-truth default flag*  
The supervised learning target. Two columns: `borrower_id` and `default_flag`. Default rate is calibrated to 18% — consistent with Mexican micro-credit portfolio default rates reported by Compartamos Banco and CNBV. The label is derived from a latent risk model combining borrower attributes, neighbourhood graph risk, group cohesion, and behavioural signals, then calibrated via binary search to hit the 18% target rate.

---

### How the Files Work Together for Credit Scoring

```
APPLICATION TIME (what the officer observes)
┌─────────────────────────────────────────────────────────┐
│  borrowers ──► groups                                   │
│  (identity, demographics, income, liabilities)          │
└─────────────────────────┬───────────────────────────────┘
                          │
BEHAVIOURAL DATA (alternative credit signals)
┌─────────────────────────▼───────────────────────────────┐
│  cdr           (call regularity, communication volume)  │
│  mobile_events (app usage, location, digital activity)  │
│  edges         (peer network, social trust graph)       │
└─────────────────────────┬───────────────────────────────┘
                          │
LOAN HISTORY (if prior borrower)
┌─────────────────────────▼───────────────────────────────┐
│  loans + repayments  (amounts, latency, on-time rate)   │
└─────────────────────────┬───────────────────────────────┘
                          │
FEATURE ENGINEERING (Silver Layer)
┌─────────────────────────▼───────────────────────────────┐
│  features.parquet  (38 features, one row per borrower)   │
│  + gnn_embeddings  (graph position vectors)             │
└─────────────────────────┬───────────────────────────────┘
                          │
MODEL SCORING
┌─────────────────────────▼───────────────────────────────┐
│  XGBoost baseline  ──┐                                  │
│  GraphSAGE GNN     ──┼──► Hybrid Ensemble ──► labels   │
│  Federated model   ──┘         (default_flag)           │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Snowflake Schema

**Schema type used: Snowflake**

A pure star schema would require every dimension to connect directly to the central fact table. That is not possible here because `repayments` has no direct relationship with `borrowers` — a repayment belongs to a specific `loan`, not to a borrower directly. The `repayments → loans → borrowers` chain is a two-level hierarchy, which by definition makes this a snowflake. All other dimensions do connect directly to `borrowers`, making the schema mostly star-like except for that one normalised branch.

```
                         ┌──────────────────────────┐
                         │       groups (DIM)        │
                         │──────────────────────────│
                         │ PK: group_id              │
                         │     cycle_number          │
                         │     cohesion_score        │
                         └────────────┬──────────────┘
                                      │ FK: group_id
                                      │
  ┌──────────────────┐  ┌─────────────▼───────────────────────────────┐  ┌──────────────────────┐
  │    cdr (DIM)     │  │             borrowers (FACT)                 │  │  mobile_events (DIM) │
  │──────────────────│  │─────────────────────────────────────────────│  │──────────────────────│
  │ PK: borrower_id  │  │ PK: borrower_id                             │  │ PK: borrower_id      │
  │ FK: borrower_id  ├──│ FK: group_id → groups                       ├──│ FK: borrower_id      │
  │ call_volume      │  │ age                  married_flag            │  │ app_opens            │
  │ call_routine_    │  │ gender               num_children            │  │ location_variance    │
  │   score          │  │ city                 monthly_income_MXN      │  │ falabella_app_       │
  │ messaging_       │  │ rural_flag           rent_MXN               │  │   session_flag       │
  │   frequency      │  │ indigenous_proxy     mobile_phone_plan_MXN  │  │ routine_entropy      │
  │ weekly_call_cv   │  │ CURP_hash            electricity_water_MXN  │  │ CoDi_txn_regularity  │
  └──────────────────┘  │ INE_verified_flag    informal_loans_MXN     │  │ app_session_         │
                        │ store_visit_count    formal_debt_payments_  │  │   recency_days       │
                        │ prior_CMR_usage        MXN                  │  └──────────────────────┘
                        │ CoDi_wallet_flag                            │
                        └──────────┬──────────────┬───────────────────┘
                                   │              │
                        borrower_id│              │borrower_id
                                   │              │
              ┌────────────────────▼──┐     ┌─────▼──────────────┐
              │      loans (DIM)      │     │    labels (DIM)    │
              │───────────────────────│     │────────────────────│
              │ PK: loan_id           │     │ PK: borrower_id    │
              │ FK: borrower_id       │     │ FK: borrower_id    │
              │ amount_MXN            │     │ default_flag       │
              │ product_type          │     └────────────────────┘
              │ CMR_credit_line_flag  │
              │ OXXO_cash_backed_flag │
              └───────────┬───────────┘
                          │ FK: loan_id
                          │ (snowflake branch)
              ┌───────────▼───────────┐
              │   repayments (DIM)    │
              │───────────────────────│
              │ PK: loan_id           │
              │ FK: loan_id → loans   │
              │ due_date              │
              │ paid_date             │
              │ amount                │
              │ repayment_latency_days│
              └───────────────────────┘

  ┌────────────────────────────────────────┐
  │          edges (BRIDGE TABLE)          │
  │────────────────────────────────────────│
  │ PK: (src_id, dst_id)                   │
  │ FK: src_id → borrowers.borrower_id     │
  │ FK: dst_id → borrowers.borrower_id     │
  │ tie_strength                           │
  │ WhatsApp_metadata_proxy                │
  │ CoDi_transfer_link                     │
  └────────────────────────────────────────┘
  Resolves the many-to-many peer relationship
  between borrowers in the social graph.
```

### Schema Classification

| File | Type | Primary Key | Foreign Key(s) |
|---|---|---|---|
| `borrowers` | **Fact** (central table) | `borrower_id` | `group_id → groups.group_id` |
| `groups` | **Dimension** | `group_id` | — |
| `cdr` | **Dimension** | `borrower_id` | `borrower_id → borrowers.borrower_id` |
| `mobile_events` | **Dimension** | `borrower_id` | `borrower_id → borrowers.borrower_id` |
| `labels` | **Dimension** | `borrower_id` | `borrower_id → borrowers.borrower_id` |
| `loans` | **Dimension** | `loan_id` | `borrower_id → borrowers.borrower_id` |
| `repayments` | **Dimension** (snowflake branch) | `loan_id` | `loan_id → loans.loan_id` |
| `edges` | **Bridge Table** | `(src_id, dst_id)` | `src_id → borrowers.borrower_id`, `dst_id → borrowers.borrower_id` |

**Why `borrowers` is the fact table:** The analytical grain of this system is one credit applicant assessment. `borrowers` holds every numeric measure that characterises credit risk — income, liabilities, behavioural counts — and is the unit at which the model scores and labels predict. Every other file either describes the borrower further (`groups`, `cdr`, `mobile_events`, `labels`) or records historical transactions belonging to the borrower (`loans`, `repayments`).

**Why snowflake and not star:** `repayments` cannot connect directly to `borrowers` — a repayment event belongs to a specific loan, not to a borrower directly. The `repayments → loans → borrowers` two-level chain is the defining snowflake characteristic. All other dimensions connect directly to `borrowers`, so the schema is mostly flat, with one normalised branch.

---

## 5. How the Synthetic Dataset Resembles Actual Mexican Demographics

Although every record is synthetically generated, each statistical parameter was calibrated against official Mexican government sources. The table below compares key dataset metrics against their real-world counterparts.

| Dimension | Dataset Value | Real Mexico Value | Source |
|---|---|---|---|
| Gender (female) | 70.2% | ~70% (micro-credit market) | Compartamos Banco Annual Report 2022 |
| Rural population | 20.8% | 21.0% | INEGI CPV 2020 |
| Indigenous self-ID | 17.9% | 18.4% (23.2M / 126M) | INEGI CPV 2020 |
| INE credential coverage | 90.5% | ~90% of adults | INE Padrón Electoral 2022 |
| CoDi wallet adoption | 13.0% | ~12% (10.7M / ~90M adults) | Banxico CoDi stats, Sept 2021 |
| Urban income median | 8,326 MXN | ~8,000 MXN individual | INEGI ENIGH 2022 |
| Rural income median | 4,654 MXN | ~4,500 MXN individual | INEGI ENIGH 2022 |
| Renters share | 30.7% | ~30% of households | INEGI ENIGH 2022 |
| Postpaid mobile share | 30.7% | ~30% of mobile lines | IFT 2022 |
| Informal credit usage | 22.1% | ~23% of adults | ENIF 2021 (INEGI/CNBV) |
| Formal credit (thin-file adj.) | 13.8% | 32.7% national; ~13% micro-credit pool | ENIF 2021 |
| Electricity/water access | 90.3% | ~90–93% urban; ~80% rural | INEGI CPV 2020; CFE 2022 |
| Portfolio default rate | 17.94% | ~15–20% micro-credit portfolios | CNBV Inclusion Reports 2022 |
| Mean borrower age | 35.9 years | ~35–38 (micro-credit borrower) | Compartamos Banco / Gentera Mexico |

### Key Ways the Dataset Resembles Real Mexico

**1. Financial exclusion is structural, not incidental**  
72.2% of borrowers have no prior formal credit (thin-file). This reflects Mexico's real financial exclusion rate — ENIF 2021 shows 40% of adults are fully unbanked, and micro-credit institutions specifically serve the underserved base of the pyramid who have been bypassed by traditional banks.

**2. Gender composition reflects the micro-credit market**  
The 70% female composition is not an arbitrary choice — it mirrors the documented reality of solidarity lending in Mexico and Latin America. Women borrowers dominate micro-credit because they show stronger repayment discipline in group-lending contexts, a pattern confirmed by Compartamos Banco's two decades of operational data.

**3. Rural borrowers face compounding disadvantages**  
Rural borrowers in the dataset have lower income (rural median MXN 4,654 vs urban MXN 8,326), higher informal credit reliance (30% vs 20%), lower INE verification, and higher indigenous proxy rates. This mirrors documented patterns in INEGI and CNBV financial inclusion reports showing rural Mexico is persistently underserved across income, identity verification, and credit access dimensions.

**4. The social graph reflects solidarity lending reality**  
The 7,500 peer groups with cohesion scores model the real-world Grameen-style solidarity circle. Higher-cohesion groups generate denser peer networks (correlation: 0.643) — consistent with sociological research showing that tight community bonds reduce loan default through social pressure and mutual accountability.

**5. CoDi as a digital divide proxy**  
CoDi adoption at 13.0% closely mirrors Banxico's reported 10.7 million accounts against an adult population of ~90 million. The lower adoption among rural and older borrowers in the dataset replicates the documented digital divide in Mexico — urban, higher-income, younger adults adopted CoDi; rural and lower-income populations remained on prepaid cash-based systems.

**6. Income inequality follows ENIGH-calibrated log-normal distribution**  
The Gini-consistent income spread (min MXN 1,200, median MXN 7,446, max MXN 144,253) with a right-skewed distribution mirrors the ENIGH 2022 income distribution. The gender wage penalty of 14% (women earning ~86% of men) is taken directly from INEGI ECAP 2022 labour force data.

**7. Default rate calibrated to market reality**  
The 18% portfolio default rate aligns with Compartamos Banco's reported non-performing loan rates and CNBV micro-credit portfolio data, placing it within the 15–20% range considered normal for unsecured micro-lending in Mexico.

---

## Official Sources Referenced

| Source | Year | Usage |
|---|---|---|
| INEGI Censo de Población y Vivienda (CPV) 2020 | 2020 | Age, gender, rural/urban, indigenous, city populations, INE coverage |
| INEGI ENIGH (Encuesta Nacional de Ingresos y Gastos de los Hogares) 2022 | 2022 | Income distribution, rental share, household expenditure |
| INEGI ENIF (Encuesta Nacional de Inclusión Financiera) 2021 | 2021 | Formal credit rate (32.7%), informal credit usage (23%), financial inclusion metrics |
| INEGI ECAP (Encuesta de Ocupación y Empleo / gender wage data) 2022 | 2022 | Gender wage gap (~86% women earn relative to men) |
| CONAPO Proyecciones de Población 2020 | 2020 | Fertility rates (TFR 1.6 national; 2.1 rural; 1.5 urban) |
| Banxico — CoDi system statistics | Sept 2021 | CoDi adoption: 10.7M accounts / ~90M adults ≈ 12% |
| INE Padrón Electoral | 2022 | INE credential coverage ~90% of adults |
| IFT (Instituto Federal de Telecomunicaciones) | 2022 | Postpaid mobile lines ~30% of total |
| CFE (Comisión Federal de Electricidad) residential tariffs | 2022 | Monthly electricity cost benchmarks |
| Compartamos Banco Annual Report | 2022 | Female borrower share (~70%), default rate benchmarks, borrower age profile |
| CNBV Reporte de Inclusión Financiera | 2022 | Portfolio default rates, financial inclusion by region |
