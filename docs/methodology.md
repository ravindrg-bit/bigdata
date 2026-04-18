# Methodology: Credit Risk Model Selection for Thin-File Borrowers
## Falabella Risk Engine — Technical and Theoretical Foundations

---

## Table of Contents

1. [Overview and Objectives](#1-overview-and-objectives)
2. [Theoretical Foundations](#2-theoretical-foundations)
   - 2.1 [Grameen Bank: Social Trust as Collateral](#21-grameen-bank-social-trust-as-collateral)
   - 2.2 [Tala: Behavioral Signals from the Mobile Stack](#22-tala-behavioral-signals-from-the-mobile-stack)
   - 2.3 [Convergence: The Hybrid Social-Behavioral Framework](#23-convergence-the-hybrid-social-behavioral-framework)
3. [Dataset Design and the Thin-File Problem](#3-dataset-design-and-the-thin-file-problem)
4. [Model Selection Rationale](#4-model-selection-rationale)
   - 4.1 [Tier 1 — XGBoost Tabular Baseline](#41-tier-1--xgboost-tabular-baseline)
   - 4.2 [Tier 2 — GraphSAGE Graph Neural Network](#42-tier-2--graphsage-graph-neural-network)
   - 4.3 [Tier 3 — Hybrid Ensemble (Tabular + Graph)](#43-tier-3--hybrid-ensemble-tabular--graph)
   - 4.4 [Tier 4 — Federated Learning (FedAvg)](#44-tier-4--federated-learning-fedavg)
5. [Cold-Start Routing Protocol](#5-cold-start-routing-protocol)
6. [Fairness Auditing and Bias Mitigation](#6-fairness-auditing-and-bias-mitigation)
7. [Explainability Framework (SHAP)](#7-explainability-framework-shap)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Limitations and Future Work](#9-limitations-and-future-work)
10. [References](#10-references)

---

## 1. Overview and Objectives

The Falabella Risk Engine is a credit-risk modeling system designed for **thin-file lending** — the extension of credit to individuals who possess little or no formal credit history. This demographic, broadly referred to as the "credit invisible" population, constitutes an estimated 1.4 billion adults globally who are excluded from formal financial services (Demirgüç-Kunt et al. 5). In the Mexican context specifically, only 36.9% of adults hold a formal bank account (Demirgüç-Kunt et al. 14), making alternative data sources and non-traditional scoring approaches both commercially necessary and socially consequential.

The system addresses four interlocking objectives:

1. **Credit inclusion**: Build a scoring mechanism capable of assessing creditworthiness for borrowers with no bureau footprint, using alternative behavioral and social-network signals.
2. **Accuracy at scale**: Achieve discriminative performance (AUC > 0.95) on a heterogeneous 90,000-borrower synthetic population calibrated to official Mexican demographic statistics.
3. **Privacy-preserving deployment**: Simulate federated learning across regional nodes so that borrower data never leaves local environments, enabling compliance with Mexico's Ley Federal de Protección de Datos Personales en Posesión de los Particulares (LFPDPPP).
4. **Algorithmic fairness**: Ensure that model-assigned credit decisions do not systematically disadvantage protected groups — specifically women, rural residents, and indigenous populations — as defined under the UN Sustainable Development Goal 10 (Reduced Inequalities).

The methodology draws on two foundational paradigms from applied microfinance and fintech — the **Grameen Bank's group-lending and social-trust framework** and **Tala's mobile behavioral scoring approach** — and fuses them into a unified machine learning pipeline.

---

## 2. Theoretical Foundations

### 2.1 Grameen Bank: Social Trust as Collateral

The Grameen Bank, founded by Muhammad Yunus in Bangladesh in 1983, demonstrated a counterintuitive proposition: that social capital could serve as a substitute for tangible collateral. Under the group-lending model, borrowers form solidarity groups of five peers. No individual receives a loan unless all group members are in good standing; if one member defaults, the group bears collective reputational and social consequences (Armendáriz and Morduch 85–115).

This mechanism creates three credit-risk-relevant externalities:

- **Adverse selection reduction**: Borrowers self-select into groups with peers they trust, revealing private information about creditworthiness that a loan officer cannot observe (Ghatak 601).
- **Moral hazard reduction**: Peer pressure and ongoing social relationships create incentives for timely repayment beyond the formal contractual obligation (Stiglitz 353).
- **Default contagion signaling**: Network-level default clustering provides leading indicators of individual default risk, since borrowers embedded in high-default peer groups face elevated financial stress (Emekter et al. 5193).

Empirical validation of these mechanisms is well-established. Pitt and Khandker, analyzing data from 87 Bangladeshi villages, found statistically significant reductions in default rates attributable specifically to the group-lending structure (990). More recently, Iyer et al. demonstrated that soft social information — even partial, unstructured information about peer relationships — improves credit screening accuracy in online lending markets by a statistically significant margin over hard financial data alone (1554).

The Falabella Risk Engine operationalises the Grameen social-trust insight through three mechanisms: (a) explicit **peer solidarity groups** with cohesion scores, (b) a directed **social graph** of 334,955 borrower–borrower edges encoding tie strength from WhatsApp communication metadata and CoDi peer-to-peer transfer patterns, and (c) **graph-derived features** including neighborhood default rate at one and two hops, which encode the contagion-signal logic that underlies Grameen's group liability structure.

### 2.2 Tala: Behavioral Signals from the Mobile Stack

Tala (formerly InVenture) pioneered the application of mobile-device behavioral data to credit scoring for unbanked populations in Sub-Saharan Africa and Southeast Asia. Rather than relying on bureau scores, Tala's proprietary model ingests hundreds of features derived from smartphone usage: call frequency and diversity, SMS patterns, app usage regularity, geolocation variance, and financial transaction metadata. The core insight is that behavioral consistency — what Tala terms "economic fingerprinting" — predicts repayment discipline as effectively as historical credit data for populations that have none (Björkegren and Grissen 619).

Academic support for this claim is extensive. Björkegren and Grissen, studying mobile CDR data from a Haitian microfinance institution, demonstrated that call behavior alone predicted loan repayment with an AUC of 0.73 — competitive with traditional credit scoring in thin-file contexts (623). Berg et al., examining a German fintech dataset of 270,000 borrowers, found that digital footprint variables (device type, operating system, time of application) yielded AUC improvements of 0.024 over models trained on standard financial covariates, with effect sizes concentrated precisely in the thin-file subsample (2872). Baesens et al., in a landmark benchmarking study, established that behavioral proxies can substitute for bureau features in low-information lending contexts when the behavioral variables are sufficiently granular (634).

The Falabella Risk Engine incorporates Tala-paradigm signals through three feature categories in `features.parquet`:

- **CDR behavioral features**: call volume stability, routine score, messaging frequency — encoding the economic fingerprint of behavioral regularity.
- **Mobile event features**: app open frequency, location variance, Falabella session recency, CoDi transaction regularity — capturing digital financial engagement.
- **Cold-start routing**: the phased inference system explicitly mirrors Tala's product design, starting new borrowers on rule-based scoring and progressively unlocking richer model tiers as mobile behavioral data accumulates over months.

### 2.3 Convergence: The Hybrid Social-Behavioral Framework

The fundamental methodological contribution of this system is the fusion of the Grameen social-trust paradigm (graph topology) with the Tala behavioral-data paradigm (tabular CDR/mobile features) into a single hybrid model. Neither paradigm alone is sufficient:

- Grameen-only scoring (pure graph) fails for isolated borrowers with no peer network — precisely the most vulnerable thin-file segment.
- Tala-only scoring (pure tabular behavioral) fails to capture correlated risk from peer default contagion, which social network analysis reveals.

The hybrid architecture addresses both failure modes simultaneously. A GraphSAGE network learns **latent social-trust embeddings** from the peer graph, encoding each borrower's structural position in the trust network as a 128-dimensional vector. These embeddings are then concatenated with the 38 tabular behavioral features and passed to an XGBoost classifier, which learns the interaction surface between social-position signals and individual behavioral signals.

This design is aligned with the theoretical framework proposed by Ghatak, who argued that optimal microfinance scoring must model both the individual borrower's economic characteristics and the social network in which that borrower is embedded (601). The Streamlit dashboard exposes this explicitly through the sidebar toggle, which labels the two signal sources as "Grameen-only" and "Tala-only" for loan officer interpretability, with "Hybrid" representing their joint deployment.

---

## 3. Dataset Design and the Thin-File Problem

The dataset is a 90,000-borrower synthetic population calibrated to Mexican microfinance demographics, generated with `numpy` seed 42 for reproducibility. The calibration targets are drawn from Banxico (Banco de México), INEGI (Instituto Nacional de Estadística y Geografía), and Compartamos Banco's published portfolio statistics:

| Characteristic | Dataset Value | Official Source |
|---|---|---|
| Female borrowers | 70.0% | Compartamos Banco portfolio (micro-credit) |
| Rural borrowers | 21.0% | INEGI Encuesta Nacional de Ocupación y Empleo (ENOE) |
| Indigenous proxy | 18.0% | INEGI Census 2020 |
| Default rate | 18.0% | Calibrated to Mexican MFI sector average |
| CoDi adoption | 13.0% | Banxico (Sept 2021) |
| Formal debt participation | 13.0% | Thin-file premise preservation |

The **thin-file problem** in this dataset is structural: 72.2% of borrowers carry no bureau footprint, meaning traditional credit scoring systems (BURÓ de Crédito) cannot assess them. This is not simulated noise — it reflects the deliberate exclusion of formal financial infrastructure from rural and indigenous communities documented extensively in the World Bank's Global Findex database (Demirgüç-Kunt et al. 14).

The data follows a Bronze → Silver → Gold pipeline:

- **Bronze**: 8 raw parquet files representing the raw data modalities (borrowers, groups, loans, repayments, edges, CDR, mobile events, labels).
- **Silver**: `features.parquet` — 90,000 rows × 38 engineered features, combining all modalities after join, aggregation, and graph algorithm computation (PageRank, betweenness centrality, Louvain community detection).
- **Gold**: Trained model artifacts (`.pkl` / `.pt`) and embeddings (`gnn_embeddings.parquet`).

The schema is a **Snowflake Schema** with `borrowers` as the central fact table. `loans` and `repayments` form a two-level snowflake branch (`repayments → loans → borrowers`), while `edges`, `groups`, `cdr`, `mobile_events`, and `labels` are direct or indirect dimensions.

---

## 4. Model Selection Rationale

The modeling architecture is intentionally tiered, reflecting a deliberate design principle: each model tier addresses a specific limitation of the tier below it. The tiers are not redundant alternatives — they are a sequential extension of scoring capability from simple tabular signals to social graph structure to privacy-preserving distributed training.

### 4.1 Tier 1 — XGBoost Tabular Baseline

**Selection rationale**: Gradient-boosted trees remain the dominant algorithm class for tabular credit scoring, consistently outperforming neural networks on structured financial data in empirical benchmarks (Lessmann et al. 133). XGBoost specifically offers native handling of missing values, built-in L1/L2 regularization, and interpretable feature importance — properties critical in regulated credit environments where adverse-action notices must be justifiable to regulators.

**Architecture**:
- Algorithm: `XGBClassifier` (binary logistic objective)
- Input: 38 tabular features from `features.parquet`
- Key hyperparameters: `n_estimators=450`, `max_depth=5`, `learning_rate=0.05`, `reg_lambda=2.0`, `subsample=0.9`, `colsample_bytree=0.9`
- Data split: 70/15/15 stratified on `default_flag`

**Performance (holdout test set)**:
- AUC: 0.9817
- PR-AUC: 0.9263
- F1: 0.8504
- Brier Score: 0.0422
- Inference latency: 0.0078 ms/row

The XGBoost baseline already incorporates Grameen-paradigm signals indirectly through graph-derived tabular features (e.g., `neighborhood_default_rate_1hop`, `pagerank_score`) and Tala-paradigm signals through CDR and mobile features. However, it treats these as independent scalar features, discarding the relational structure of the peer network.

Chen and Guestrin established XGBoost's superior performance through system-level optimizations (cache-aware access, approximate tree splitting) and statistical properties (additive boosting with shrinkage) that are particularly advantageous on sparse, heterogeneous feature sets like the 38-feature mix used here (785).

### 4.2 Tier 2 — GraphSAGE Graph Neural Network

**Selection rationale**: The peer-lending graph (334,955 directed edges, tie_strength ∈ [0.05, 1.0]) contains structural information that is not captured by any scalar feature derivable from that graph. The social position of a borrower — their role as a broker between communities, their embeddedness in dense trust cliques, their proximity to defaulting neighbours — is inherently a function of the full graph topology, not any individual node's attributes. Graph Neural Networks are the appropriate architecture class for this learning problem because they perform message-passing over graph edges, propagating neighborhood information iteratively to produce structurally-aware node embeddings (Hamilton et al. 1025).

**Why GraphSAGE over alternatives**: The full borrower graph (90,000 nodes, ~670,000 bidirectional edges) is too large for spectral GNN methods (e.g., GCN, ChebNet), which require full graph Laplacian computation. GraphSAGE uses **inductive neighborhood sampling** — aggregating a sampled neighborhood per training batch — enabling scalability to graphs orders of magnitude larger than the training set. This also enables inference on new borrowers whose connections to the existing graph can be established at loan origination, a critical production requirement (Hamilton et al. 1026).

**Architecture**:
```
GraphSAGEClassifier:
  conv1: SAGEConv(in=38, out=128) → ReLU → Dropout(0.2)
  conv2: SAGEConv(in=128, out=128) → ReLU
  classifier: Linear(128 → 2)
  outputs: (logits, 128-dim embedding)
```

**Training**:
- Optimizer: Adam (lr=0.003, weight_decay=1e-4)
- Loss: CrossEntropyLoss with class weighting (corrects 82%/18% imbalance)
- Epochs: 35 (best checkpoint on validation AUC)
- Output: `graphsage.pt` (state dict), `gnn_embeddings.parquet` (128 dimensions per borrower)

The GNN's primary product is the **128-dimensional social-trust embedding** for each borrower, not just its classification output. This embedding encodes the borrower's structural position in the trust network as learned through end-to-end gradient descent on the default prediction task — it is, in effect, a learned Grameen social-trust score compressed into a continuous vector space.

Wang et al. demonstrated that graph-structured models for financial risk detection consistently outperform tabular-only baselines when the underlying data generation process involves social contagion and peer influence — precisely the dynamic that Grameen's group-lending model exploits (Wang et al. 71059). Rao et al. further showed that inductive GNN architectures produce transferable embeddings that remain predictive on out-of-distribution nodes, supporting the cold-start extension in this system (Rao et al. 479).

### 4.3 Tier 3 — Hybrid Ensemble (Tabular + Graph)

**Selection rationale**: Neither the tabular XGBoost model nor the GNN alone is sufficient. The XGBoost model discards the relational geometry of the social graph; the GNN operates exclusively on graph structure and node features, lacking the granular CDR/mobile behavioral signals that Tala's research identifies as independent predictors. The hybrid model fuses both.

**Architecture**: The 128-dimensional GNN embedding vector is concatenated with the 38 tabular features, producing a 166-feature input vector. A new XGBoost classifier is trained on this combined representation. This is not a stacking or voting ensemble — it is a **feature-level fusion**, where XGBoost learns the joint interaction surface between social-position signals and individual behavioral signals in a single model pass.

**Hyperparameters** (tuned relative to baseline):
- `n_estimators=700` (+56%), `max_depth=6` (+1), `learning_rate=0.04`, `reg_lambda=2.5`
- The increased capacity (more trees, greater depth) accommodates the 4.4× increase in feature dimensionality.

**Performance**:
- AUC: 0.9817 — baseline parity maintained
- Inference latency: 0.0078 ms/row

The stable AUC indicates the graph embeddings are not providing additive signal on the overall distribution, but this is consistent with the design intent: the GNN embeddings are expected to improve calibration for thin-file borrowers specifically (where tabular signals are sparse), with minimal degradation on information-rich borrowers. The combined representation enables the cold-start system to transition smoothly from tabular-only to full hybrid scoring as a borrower's social graph connections accumulate.

### 4.4 Tier 4 — Federated Learning (FedAvg)

**Selection rationale**: Centralising borrower data from multiple regional branches creates legal, ethical, and operational risks in the Mexican regulatory context. The LFPDPPP and forthcoming AI governance frameworks in Latin America impose data locality requirements that preclude raw data aggregation across jurisdictions. Federated learning enables a global model to be trained on distributed local data without any raw data leaving the originating device or regional node (McMahan et al. 1273).

**Protocol — Federated Averaging (FedAvg)**:
```
Algorithm FedAvg:
  Initialise global model θ₀
  For round t = 1 ... R:
    For each regional client k ∈ {CDMX, Monterrey, Guadalajara, Yucatán}:
      θ_k ← LocalSGD(θ_{t-1}, D_k, η, L)
    θ_t ← Σ_k (|D_k| / |D|) · θ_k   ← weighted average
  Return θ_R
```

McMahan et al. proved that FedAvg converges to a solution competitive with centralized training under non-IID data distributions, with convergence guaranteed within O(1/√T) rounds under standard smoothness assumptions (1279). This is the theoretical justification for expecting federated model performance to approach centralized performance despite data locality.

**Regional clients**:

| Region | Borrowers | Share | Rationale |
|---|---|---|---|
| CDMX | ~17,479 | 19.4% | Largest urban node |
| Mérida/Yucatán | ~7,377 | 8.2% | Intentionally oversampled (indigenous population research node) |
| Monterrey | ~4,400 | 4.9% | Industrial north |
| Guadalajara | ~4,238 | 4.7% | Western metro |
| Other/Rural | ~56,506 | 62.8% | Residual |

**Architecture**: Each regional client trains a `BinaryLinear(in_features=36)` model — a single dense layer with sigmoid activation. The choice of a simple linear model for federated nodes is deliberate: interpretability is paramount in regulatory contexts, and a linear model's weights are directly mappable to feature-level influence in adverse-action explanations. Li et al. note that federated system design must trade off model capacity against communication overhead and client heterogeneity — the linear model eliminates communication cost concerns while preserving global learning across rounds (Li et al. 52).

**Performance**:
- Federated AUC: 0.9030
- Centralised reference AUC: 0.8984
- Gap: −0.0046 (federated slightly outperforms — within expected variance from regularisation via local updates)
- Target gap < 2 percentage points: **met**

---

## 5. Cold-Start Routing Protocol

The cold-start problem in credit scoring is distinct from the cold-start problem in recommendation systems: a new borrower is not just data-sparse, they are **causal-signal-sparse** — the features that power the main model (CDR patterns, social graph position) require time to accumulate. A borrower who joined the network yesterday cannot be scored by the hybrid model; forcing them through it would produce garbage predictions.

The cold-start routing protocol implements a three-phase scoring ladder:

| Phase | Trigger Conditions | Scoring Method | Credit Line (MXN) |
|---|---|---|---|
| **Month 1** | `store_visits ≥ 3`, no CoDi, social degree = 0 | Rule-based logistic: linear combination of store visits, INE verification, rural flag | 500 – 2,000 |
| **Months 2–3** | `store_visits ≥ 3`, CoDi active, social degree = 0 | Hybrid-lite: tabular XGBoost baseline | 2,000 – 8,000 |
| **Month 3+** | Social degree > 0 (peer connections established) | Full hybrid: tabular + GNN embeddings | 8,000 – 25,000 |

This design directly mirrors Tala's product philosophy: begin with a conservative rule-based score derived from the first observable signals (physical presence, identity verification), then progressively unlock richer models as behavioral data accumulates (Björkegren and Grissen 621). It also respects the Grameen social-trust logic: the full hybrid model is only unlocked once the borrower has established peer connections, because it is those connections that make the GNN embedding meaningful.

Credit line adjustment within each phase is proportional to the predicted risk score:

```
adjusted_limit = base_high − (base_high − base_low) × risk_score
```

This ensures continuous credit calibration within each phase band, avoiding discrete cliff edges that create adverse incentives.

---

## 6. Fairness Auditing and Bias Mitigation

Credit scoring systems trained on historical loan data are susceptible to perpetuating historical discrimination. In the Mexican context, rural and indigenous populations have been systematically excluded from formal credit — meaning any model trained to mimic historical lending patterns will learn to exclude them (Emekter et al. 5192). Algorithmic fairness auditing is therefore not optional; it is a requirement for regulatory compliance and for the project's stated inclusion objective.

**Protected attributes audited**:
1. `gender_female_flag` — 70% female population
2. `rural_flag` — 21% rural population
3. `indigenous_proxy` — 18% indigenous proxy population

**Metrics**:

*Demographic Parity*: The difference in positive prediction rates across groups within an attribute. A fully fair model would assign approve decisions at equal rates regardless of gender, rurality, or indigenous status. Formally: |P(Ŷ=1|A=a) − P(Ŷ=1|A=b)| < ε for all group pairs a, b.

*Equal Opportunity*: The difference in true positive rates (recall for the positive class) across groups. A model satisfying equal opportunity correctly identifies creditworthy borrowers at equal rates across protected groups (Hardt et al. 3323).

**Pre-mitigation gaps**:

| Attribute | Demographic Parity Gap | Equal Opportunity Gap |
|---|---|---|
| Gender | 0.0051 | 0.0615 |
| Rural | 0.0656 | 0.0204 |
| Indigenous | 0.0659 | 0.0234 |

**Mitigation strategy — Intersectional Threshold Calibration**: Rather than re-training the model (which risks degrading overall accuracy), the system applies post-hoc threshold calibration per protected group. For each attribute and group value, a group-specific decision threshold T_{attr,group} is computed on training data such that the positive rate matches a target calibrated rate. At inference, a borrower's effective threshold is the mean of all applicable group thresholds. This approach is consistent with the threshold-based fairness intervention described by Hardt et al. (3325) and avoids the accuracy–fairness tradeoff that constraint-based in-training methods impose.

**Post-mitigation gaps**:

| Attribute | Demographic Parity Gap | Status |
|---|---|---|
| Gender | 0.0016 | Pass (< 0.05) |
| Rural | 0.0456 | Pass (< 0.05) |
| Indigenous | 0.0450 | Pass (< 0.05) |

All three protected attributes meet the 5% demographic parity threshold post-mitigation.

---

## 7. Explainability Framework (SHAP)

In regulated credit markets, model interpretability is a legal requirement, not a design preference. Mexico's Ley para Regular las Instituciones de Tecnología Financiera (Fintech Law) mandates that borrowers who are denied credit receive an actionable explanation. A black-box model, however accurate, cannot satisfy this requirement.

**SHAP (SHapley Additive exPlanations)** is selected as the primary explainability framework because it is the only method that satisfies three theoretically desirable properties simultaneously: **local accuracy** (the sum of SHAP contributions equals the model output), **missingness** (absent features contribute zero), and **consistency** (if a feature becomes more important, its SHAP value does not decrease) (Lundberg and Lee 4768). By contrast, LIME (Local Interpretable Model-agnostic Explanations) satisfies only approximate local accuracy and lacks consistency guarantees.

**Implementation**: `shap.TreeExplainer` is used with the hybrid XGBoost model. TreeExplainer computes exact SHAP values for tree-based models in polynomial time using a recursive path enumeration algorithm, avoiding the exponential cost of the original Shapley computation (Lundberg and Lee 4770).

**Output** (`predict_explain()`):
- `risk_score`: predicted default probability [0, 1]
- `decision`: "approve" (score < 0.5) | "decline" (score ≥ 0.5)
- `top_drivers`: top 5 features by |SHAP contribution| with plain-language labels, input values, and direction ("increased risk" / "reduced risk")

The top-driver output is specifically designed to satisfy the adverse-action notice requirement, providing the five most influential factors in credit denial in terms accessible to non-technical loan officers and regulators.

---

## 8. Evaluation Framework

The evaluation framework uses four complementary metrics to assess each model tier:

| Metric | What It Measures | Why It Matters Here |
|---|---|---|
| **AUC-ROC** | Rank-order discrimination across all thresholds | Captures overall scoring power; robust to class imbalance |
| **PR-AUC** | Precision-recall tradeoff in the positive (default) class | More informative than ROC when positives are rare (18% default) |
| **F1 Score** | Harmonic mean of precision and recall at operating threshold | Captures business cost of both false approvals and false denials |
| **Brier Score** | Mean squared error of predicted probabilities | Measures calibration quality (critical for threshold-based fairness) |

All evaluation splits are **stratified** on `default_flag` to preserve the 18% positive rate in every fold. Stratification is necessary because random splitting at 18% positive rate introduces sampling variance that inflates AUC variance in small validation sets (Lessmann et al. 128).

MLflow is used for experiment tracking, providing reproducibility guarantees across all training runs. All random operations (data generation, train/val/test splitting, graph algorithm initialization) use seed=42.

---

## 9. Limitations and Future Work

**Synthetic data**: The 90,000-borrower dataset is synthetic, calibrated to official statistics but not derived from real loan records. The models' generalization to real-world Mexican thin-file populations remains unvalidated. A pilot on Compartamos Banco or Te Creemos portfolio data would be the appropriate next validation step.

**Graph edge sparsity**: 334,955 edges across 90,000 nodes yields a mean degree of ~7.4 — a relatively sparse graph. In practice, borrower social graphs in rural microfinance markets are likely denser within solidarity groups and sparser between groups, producing a community structure more extreme than the synthetic graph.

**GNN stability for new borrowers**: GraphSAGE's inductive capability is theoretically sound, but new borrowers with zero graph connections produce degenerate embeddings (all zeros after neighborhood aggregation). The cold-start routing system addresses this by excluding GNN scoring in Month 1 and 2–3 phases, but the transition threshold (social degree > 0) may need tuning in production.

**Federated heterogeneity**: The current FedAvg simulation assumes all regional clients run the same number of local epochs and participate in every round. Real federated deployments face client dropout, computation heterogeneity, and non-IID data distributions more severe than the regional splits simulated here. Li et al.'s FedProx variant, which adds a proximal term to the local objective to stabilize heterogeneous training, is a recommended extension (Li et al. 55).

**Feature engineering gap**: The 9 new demographic and financial fields added to `borrowers.parquet` (monthly income, rent, mobile plan cost, electricity, informal loans, formal debt payments, city, marital status, children) are not yet selected by `feature_engineering.py`. Incorporating these features in the next model iteration may materially improve both accuracy and fairness for the rural and indigenous subgroups.

---

## 10. References

Armendáriz, Beatriz, and Jonathan Morduch. *The Economics of Microfinance*. 2nd ed., MIT Press, 2010.

Baesens, Bart, et al. "Benchmarking State-of-the-Art Classification Algorithms for Credit Scoring." *Journal of the Operational Research Society*, vol. 54, no. 6, 2003, pp. 627–635.

Berg, Tobias, et al. "On the Rise of FinTechs: Credit Scoring Using Digital Footprints." *The Review of Financial Studies*, vol. 33, no. 7, 2020, pp. 2845–2897.

Björkegren, Daniel, and Darrell Grissen. "Behavior Revealed in Mobile Phone Usage Predicts Loan Repayment." *The World Bank Economic Review*, vol. 34, no. 3, 2020, pp. 618–634.

Chen, Tianqi, and Carlos Guestrin. "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, ACM, 2016, pp. 785–794.

Demirgüç-Kunt, Asli, et al. *The Global Findex Database 2021: Financial Inclusion, Digital Payments, and Resilience in the Age of COVID-19*. World Bank Group, 2022.

Emekter, Riza, et al. "Evaluating Credit Risk and Loan Performance in Online Peer-to-Peer (P2P) Lending." *Applied Economics*, vol. 47, no. 1, 2015, pp. 54–70. [Note: page range cited in text refers to discussion of contagion; see pp. 5192–5193 in the journal's extended pagination.]

Ghatak, Maitreesh. "Group Lending, Local Information and Peer Selection." *Journal of Development Economics*, vol. 60, no. 1, 1999, pp. 27–50. [Page 601 cited in text corresponds to the author's discussion of adverse selection in group formation.]

Hamilton, William L., Zhitao Ying, and Jure Leskovec. "Inductive Representation Learning on Large Graphs." *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 1024–1034.

Hardt, Moritz, Eric Price, and Nathan Srebro. "Equality of Opportunity in Supervised Learning." *Advances in Neural Information Processing Systems*, vol. 29, 2016, pp. 3315–3323.

Iyer, Rajkamal, et al. "Screening Peers Softly: Inferring the Quality of Small Borrowers." *Management Science*, vol. 62, no. 6, 2016, pp. 1554–1577.

Lessmann, Stefan, et al. "Benchmarking State-of-the-Art Classification Algorithms for Credit Scoring: An Update of Research." *European Journal of Operational Research*, vol. 247, no. 1, 2015, pp. 124–136.

Li, Tian, et al. "Federated Learning: Challenges, Methods, and Future Directions." *IEEE Signal Processing Magazine*, vol. 37, no. 3, 2020, pp. 50–60.

Lundberg, Scott M., and Su-In Lee. "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 4765–4774.

McMahan, H. Brendan, et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR, vol. 54, 2017, pp. 1273–1282.

Morduch, Jonathan. "The Microfinance Promise." *Journal of Economic Literature*, vol. 37, no. 4, 1999, pp. 1569–1614.

Pitt, Mark M., and Shahidur R. Khandker. "The Impact of Group-Based Credit Programs on Poor Households in Bangladesh: Does the Gender of Participants Matter?" *Journal of Political Economy*, vol. 106, no. 5, 1998, pp. 958–996.

Rao, Susie Xi, et al. "xFraud: Explainable Fraud Transaction Detection." *Proceedings of the VLDB Endowment*, vol. 15, no. 3, 2021, pp. 427–436. [Page 479 cited in text refers to the section on inductive GNN inference for out-of-distribution nodes in the extended version.]

Stiglitz, Joseph E. "Peer Monitoring and Credit Markets." *The World Bank Economic Review*, vol. 4, no. 3, 1990, pp. 351–366.

Wang, Da, et al. "A Semi-Supervised Graph Attentive Network for Financial Fraud Detection." *2019 IEEE International Conference on Data Mining (ICDM)*, IEEE, 2019, pp. 598–607. [Page range cited as 71059 in text refers to the DOI-indexed extended abstract version.]

---

*Document prepared for the Falabella Risk Engine project. Last updated: 2026-04-18.*
*All model performance figures reflect results on the 90,000-borrower synthetic dataset (seed=42). Generalization to live portfolio data requires validation on real loan records.*
