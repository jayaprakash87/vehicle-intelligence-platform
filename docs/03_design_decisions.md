# Design Decision Log

Architecture Decision Records (ADRs) for choices that shaped the system. For system structure, see [01_architecture.md](01_architecture.md). For build phases and interfaces, see [02_implementation_plan.md](02_implementation_plan.md).

---

## DD-001: Unsupervised Anomaly Detection First

**Context:** Real vehicle deployments have no labeled fault data on day one. The MVP must demonstrate anomaly detection without requiring labels.

**Decision:** Use Isolation Forest (unsupervised) as the primary anomaly model. A rules-based classifier runs in parallel for interpretability.

**Alternatives rejected:**
- Supervised (Random Forest / XGBoost) — requires curated labels that don't exist in production.
- Autoencoders — heavier dependency (PyTorch), longer training, harder to explain.

**Consequences:** Model cannot distinguish fault types on its own — that responsibility falls to the rules classifier. Accuracy improves as contamination parameter is tuned per deployment.

---

## DD-002: Rules-Based Fault Classifier

**Context:** Anomaly score alone tells you *something is wrong* but not *what* is wrong. Automotive diagnostics require fault-type identification (similar to DTC codes).

**Decision:** Threshold-based rules map feature patterns to 7 known fault types. Rules are evaluated in priority order — first match wins.

**Alternatives rejected:**
- Decision tree trained on synthetic labels — overfits to simulator artefacts, fragile when data distribution shifts.
- Multi-class classifier — needs balanced labelled data per fault type.

**Consequences:** Rules need manual tuning per vehicle platform. Thresholds are currently hardcoded in `RulesFaultClassifier.classify()`; making them config-driven is a natural next step.

---

## DD-003: DataFrame as Pipeline Currency

**Context:** Pipeline stages need a common data format. Options: raw dicts, Pydantic model lists, numpy arrays, DataFrames.

**Decision:** `pd.DataFrame` is passed between all pipeline stages.

**Trade-offs:**
- Rolling window operations, groupby, and vectorized math are native to pandas.
- Not ideal for single-row streaming — EdgeRuntime works around this by accumulating a buffer DataFrame and recomputing features over the tail window.
- At MVP data volumes (10k–100k rows), pandas is fast enough. Polars or Spark would be premature.

---

## DD-004: Config-Driven Scenario Generation

**Context:** Simulation must be reproducible, shareable, and versionable. Hardcoded parameters in code make it impossible to compare scenarios.

**Decision:** All simulation parameters (channels, faults, timing, seeds) live in YAML config files. Three scenario configs ship with the MVP.

**Consequences:** Adding a new fault scenario requires only a new YAML file — no code changes.

---

## DD-005: No Deep Learning in MVP

**Context:** Edge deployment targets resource-constrained devices (Jetson class). Model must be fast to train and tiny to deploy.

**Decision:** Exclude PyTorch/TensorFlow. IsolationForest + rules cover all MVP use cases.

**Revisit when:** Fault classification accuracy is insufficient with rules, or time-series forecasting (remaining useful life prediction) is added.

---

## DD-006: Edge Runtime = Same Code Path

**Context:** Maintaining separate batch and streaming inference paths leads to divergence and double the bugs.

**Decision:** `EdgeRuntime` is a thin loop wrapper around `InferencePipeline`. Same feature engine, same anomaly model, same classifier — called incrementally with mini-batches.

**Trade-off:** Feature recomputation over the buffer tail is redundant. Acceptable for MVP; optimize later with incremental feature computation.

---

## DD-007: Parquet as Default Storage

**Context:** Telemetry is columnar time-series data. Storage format affects query speed, file size, and schema preservation.

**Decision:** Default output format is Parquet (via pyarrow). CSV and JSON available as fallbacks.

**Rationale:** Parquet is columnar, compressed (~10x smaller than CSV), preserves dtypes, and is natively supported by pandas and Spark for downstream analysis.

---

## DD-008: Missing-Rate Tracking Before Forward-Fill

**Context:** The normalizer forward-fills NaN values to produce clean data for the feature engine. But NaN patterns carry diagnostic signal — `dropped_packet` faults produce NaN bursts that become invisible after ffill.

**Decision:** Compute a `missing_rate` column (rolling NaN ratio per channel) *before* forward-filling. This column is preserved through the pipeline and used by the rules classifier for `dropped_packet` detection.

**Consequence:** The normalizer has dual responsibility — data cleaning and feature extraction. Acceptable because the `missing_rate` is a raw data property, not a derived feature.

---

## DD-009: AI Model Evolution Roadmap

**Context:** The MVP uses Isolation Forest (unsupervised) + rules-based classifier. A natural question is why deep learning or more advanced AI models aren't used. This decision documents the constraints and the planned upgrade path.

**Why not deep learning in the MVP:**

- **No labeled data at deployment.** Supervised models (CNNs, LSTMs, transformers) need labeled fault examples. In real vehicles on day one, those don't exist. Isolation Forest is unsupervised by design.
- **Edge hardware constraints.** A Jetson Nano has 4 GB RAM and limited GPU throughput. The IsolationForest model artifact is ~100 KB. An LSTM or transformer would be 10–100x larger with 10–50x slower inference per batch.
- **Interpretability requirement.** Automotive diagnostics demand explainable outputs (OBD-II DTC codes, root cause traces). The rules classifier produces human-readable causes and recommended actions. A neural network produces a float.
- **Data volume.** The MVP generates 10k–100k rows. Deep learning models would overfit at this scale without heavy regularization or pretraining.

**Planned evolution:**

| Phase | Model | Trigger to Adopt |
|-------|-------|------------------|
| MVP (current) | IsolationForest + rules | No labels available; prove pipeline end-to-end |
| V2 | LSTM autoencoder | Enough unlabeled time-series to learn normal signal patterns; detect novel anomalies that IF misses |
| V3 | Supervised classifier (XGBoost / LightGBM) | Accumulated production labels from engineer-reviewed alerts; improves fault-type precision |
| V4 | Temporal transformer | Fleet-scale data across vehicle models; remaining-useful-life prediction; attention maps for explainability |

**Decision:** Ship MVP with IsolationForest + rules. The `AnomalyDetector` and `InferencePipeline` interfaces are model-agnostic — swapping in a new model requires implementing `train()`, `score()`, and `predict()` with no pipeline changes.

---

## Open Questions

- Should the rules classifier thresholds be config-driven instead of hardcoded?
- Target edge hardware (Jetson Nano vs Xavier NX vs Orin) — affects batch size and buffer limits.
- MQTT or HTTP for alert publishing from edge to backend?
- Should a Streamlit dashboard ship with the MVP for visual demo?
