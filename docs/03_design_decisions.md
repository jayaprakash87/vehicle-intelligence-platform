# Design Decision Log

Architecture Decision Records (ADRs) for choices that shaped the system. For system structure, see [01_architecture.md](01_architecture.md). For interface contracts and CLI reference, see [02_implementation_plan.md](02_implementation_plan.md).

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

---

## DD-010: eFuse Catalog and Vehicle Topology Factory

**Context:** The original MVP used flat channel definitions (3 channels with manually specified parameters). Scaling to realistic vehicle configurations (50+ channels) with this approach requires duplicating hundreds of lines of YAML and risks inconsistent electrical parameters.

**Decision:** Introduce a two-layer abstraction:
1. **eFuse Catalog** (`EFUSE_CATALOG`) — 17 IC family profiles (Infineon + STMicroelectronics, 2A through 100A, plus CUSTOM) with validated electrical and thermal defaults (R_ds_on, R_thermal, τ, nominal/max current).
2. **Vehicle Topology Factory** — `example_topology()` returns a declarative 4-zone, 65-channel configuration. `build_channels()` expands compact channel specs into full `ChannelMeta` by inheriting catalog defaults and zone-level settings.

**Alternatives rejected:**
- Flat YAML with all 65 channels fully specified — massive duplication, easy to introduce inconsistent R_ds_on values for the same IC family.
- Database-backed catalog — over-engineered for an MVP; YAML + Python dict is sufficient.

**Consequences:** Adding a new vehicle topology (SUV, truck) requires only a new factory function that returns zone controllers and channel specs. The catalog is reused across all topologies. Per-channel overrides still work for special cases.

---

## DD-011: Multi-Rate Protocol Support (CAN vs XCP)

**Context:** Production vehicles use CAN bus at 50–100ms per channel. Test bench measurements use XCP (Universal Measurement and Calibration Protocol) with dual-raster DAQ — 10ms for fast signals (current, voltage) and 50ms for slow signals (temperature, status). The feature engine originally assumed a single global sample rate.

**Decision:** Support per-channel `sample_interval_ms` in `ChannelMeta`. Add `SourceProtocol` enum (CAN, XCP, REPLAY). `FeatureConfig.resolve(sample_interval_s)` converts time-domain window settings to sample counts automatically. `NormalizerConfig` adds `resample_interval_ms` for optional common-grid alignment.

Transport layer: `XcpTransport` tags rows with `source_protocol=XCP` and simulates dual-raster timing. `CanTransport` tags with `source_protocol=CAN`.

**Alternatives rejected:**
- Single global rate with downsampling — loses the high-resolution XCP data that's the whole point of test bench measurement.
- Separate pipelines per protocol — duplicates the entire feature engine and inference path.

**Consequences:** The same config YAML works for both CAN and XCP data. Feature windows auto-scale (a 5s window = 50 samples at 100ms CAN, 500 samples at 10ms XCP). Resampling to a common grid is optional and configurable.

---

## DD-012: Physics-Based Signal Generation

**Context:** The original simulator used simple additive noise and step-function fault injection. This produces signals that are easy to detect but don't stress the pipeline or classifiers in ways that real vehicle data would.

**Decision:** Replace with physics-based models:
- **First-order RC thermal** — junction temperature from I²R power dissipation with device-specific R_thermal and τ.
- **Composite noise** — pink (1/f^α), ADC quantization, thermal, and sporadic EMI.
- **Load-specific inrush** — motor (5×, 50ms), inductive (3×, 20ms), PTC (2×, 500ms).
- **eFuse protection cycle** — trip → off → cooldown → retry → latch-off with realistic timing.
- **Bus voltage** — 13.5V nominal + alternator ripple + slow drift.
- **Fault envelope shaping** — trapezoidal rise/fall and damped oscillation.

**Alternatives rejected:**
- Statistical replay from recorded data — requires OEM partnership and NDA-protected datasets.
- Simple Gaussian noise + step faults — too easy for classifiers, doesn't exercise edge cases.

**Consequences:** Synthetic data now exercises the full dynamic range of the classifier. Thermal drift faults require the feature engine to track `temperature_slope` over realistic time constants. Protection cycling tests the `trip_frequency` and `recovery_time_s` features. Signal fidelity is high enough for stakeholder demos without real vehicle data.

---

## DD-013: Structured Logging with Correlation IDs

**Context:** Plain-text log output works for development but is unusable with log aggregation systems (ELK, Datadog). When running multiple pipeline commands in sequence or in parallel, there's no way to correlate log lines from the same run.

**Decision:** All log records include a `run_id` correlation ID (format: `YYYYMMDD-HHMMSS-xxxx`, stored in a `ContextVar`). Two output modes: `_PrettyFormatter` for interactive use (prefixes `[run_id]`), `_JSONFormatter` for aggregation (one JSON object per line with `ts`, `level`, `logger`, `msg`, `run_id` keys). `--json-log` CLI flag switches to JSON mode.

**Alternatives rejected:**
- OpenTelemetry — heavy dependency for an MVP; correct long-term choice when distributed tracing is needed.
- Log files per run — adds file management complexity; JSON to stdout is simpler and composable with `jq`, `tee`, etc.

**Consequences:** Every CLI run can be traced through aggregation pipelines. The `run_id` also stamps the output directory (`<output>/<run_id>/`), so log correlation maps 1:1 to on-disk artifacts.

---

## DD-014: Run-ID Output Isolation

**Context:** Sequential runs wrote to the same `output/` directory, silently overwriting `telemetry.parquet`, `scored.parquet`, etc. Results from previous runs were lost without warning.

**Decision:** Every CLI run creates `<output_dir>/<YYYYMMDD-HHMMSS-xxxx>/`. The run ID is the same correlation ID used in logging. Runs never overwrite each other.

**Alternatives rejected:**
- Append run number (run_001, run_002) — requires scanning the output directory for the latest number, race condition with parallel runs.
- Prompt before overwrite — blocks non-interactive usage (CI, scripts).

**Consequences:** Output directories accumulate. Users are responsible for cleanup. The timestamped format makes it obvious which run is which.

---

## DD-015: Edge Runtime Hardening

**Context:** The original edge loop was a simple `while not exhausted` loop with no error handling, no monitoring, and no graceful shutdown. This is fine for demos but unusable for production deployment on embedded hardware.

**Decision:** Add six hardening features:
1. **Alert rate-limiting** — suppress duplicate channel+fault alerts within a configurable cooldown window (`alert_cooldown_s`).
2. **Heartbeat** — write `heartbeat.json` every N iterations for external watchdog monitoring.
3. **Model hot-reload** — detect model file mtime change on disk, reload without restarting the loop.
4. **Signal handling** — SIGINT/SIGTERM → set `_running=False` for graceful shutdown after current batch.
5. **Error resilience** — track consecutive errors, crash only after a threshold (`max_consecutive_errors`).
6. **Disk protection** — skip writes when free space drops below `disk_min_free_mb`.

All configurable via `EdgeConfig`.

**Alternatives rejected:**
- External process supervisor (systemd, supervisord) for restart — still needed in production, but the runtime should handle transient errors itself rather than crash-looping.
- Alert deduplication in a downstream service — adds latency and infrastructure; throttling at the source is simpler for edge deployment.

**Consequences:** EdgeRuntime is now heavier (~300 lines). The added complexity is justified because edge deployment is the production path. All hardening features are tested independently.
