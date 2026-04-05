# Architecture

## System Context

This platform processes eFuse telemetry signals from vehicle electrical systems. Telemetry is synthetically generated using physics-based models that replicate real IC behavior — thermal response, protection cycling, load transients, and composite noise. The architecture supports two operating modes — **batch** (train models and run offline analysis) and **edge** (incremental scoring with real-time alerting and production hardening).

### Vehicle Electrical Architecture

```
Zone Controller (ECU)          Zone Controller (ECU)
┌──────────────────────┐       ┌──────────────────────┐
│  CDD (AUTOSAR SW)    │       │  CDD (AUTOSAR SW)    │
│  ┌─────┬─────┬─────┐ │       │  ┌─────┬─────┬─────┐ │
│  │eFuse│eFuse│eFuse│ │       │  │eFuse│eFuse│eFuse│ │
│  │HS-5A│HS-10│HS-20│ │       │  │LS-5A│HS-15│HS-50│ │
│  └──┬──┴──┬──┴──┬──┘ │       │  └──┬──┴──┬──┴──┬──┘ │
│     │SPI  │SPI  │SPI  │       │     │SPI  │SPI  │SPI  │
└─────┼─────┼─────┼────┘       └─────┼─────┼─────┼────┘
      │     │     │                   │     │     │
  Headlamp Wiper  Horn          Int.Light Mirror  HVAC
      ↓     ↓     ↓                   ↓     ↓     ↓
         CAN Bus (50-100ms)             CAN Bus (50-100ms)
              ↘                        ↙
               Gateway / XCP (10ms test)
                       ↓
              Telemetry Pipeline (this platform)
```

**Key distinction:** Zone Controllers are physical ECUs. The CDD (Complex Device Driver) is the AUTOSAR software layer inside each Zone Controller that reads eFuse registers over SPI. CAN carries production telemetry at 50–100ms. XCP is used only on test vehicles with 10ms + 50ms dual-raster DAQ.

## Pipeline Overview

```mermaid
flowchart LR
    subgraph Data Generation
        CAT[eFuse Catalog<br/>17 IC families] --> TOPO[Vehicle Topology<br/>4 zones · 65 channels]
        CFG[YAML Config] --> SIM[Simulator]
        TOPO --> SIM
        SIM --> RAW["Raw Telemetry<br/>(DataFrame)"]
        SIM --> LBL["Event Labels<br/>(ground truth)"]
    end

    subgraph Signal Processing
        RAW --> NRM[Normalizer<br/>resample · ffill · clip]
        NRM --> CLN["Cleaned Telemetry"]
        CLN --> FE[Feature Engine<br/>time-based windows]
        FE --> FEAT["Featured Telemetry<br/>(+10 derived columns)"]
    end

    subgraph Intelligence
        FEAT --> AD[Anomaly Detector<br/>Isolation Forest]
        FEAT --> RC[Rules Classifier<br/>7 fault types]
        AD --> SCORE[anomaly_score + is_anomaly]
        RC --> CLASS[fault_type + confidence + causes]
    end

    subgraph Output
        SCORE --> STR["Storage<br/>(run-ID dir)"]
        CLASS --> STR
        SCORE --> ALT[Alerts]
        CLASS --> ALT
    end
```

## Operating Modes

### Batch Mode

Used for training and offline analysis. The CLI `pipeline` command runs this end-to-end.

```mermaid
sequenceDiagram
    participant CLI
    participant Simulator
    participant Normalizer
    participant FeatureEngine
    participant AnomalyDetector
    participant RulesClassifier
    participant Storage

    CLI->>Simulator: generate(config)
    Simulator-->>CLI: telemetry_df, labels_df
    CLI->>Normalizer: normalize(telemetry_df)
    Normalizer-->>CLI: cleaned_df
    CLI->>FeatureEngine: compute(cleaned_df)
    FeatureEngine-->>CLI: featured_df
    CLI->>AnomalyDetector: train(featured_df)
    CLI->>AnomalyDetector: score(featured_df)
    AnomalyDetector-->>CLI: anomaly_scores
    CLI->>RulesClassifier: classify_df(featured_df)
    RulesClassifier-->>CLI: fault_types
    CLI->>Storage: write_scored(result_df)
```

### Edge Mode

Used for incremental inference. The `EdgeRuntime` maintains a rolling buffer, scores each mini-batch in context, and emits alerts for anomalies. Hardened with alert rate-limiting, heartbeat monitoring, model hot-reload, signal handling, and disk space protection.

Current implementation note: edge inference is driven by incoming row batches from the transport, not by a fixed 60-second timer. In the current codebase, the implemented layer is the fast streaming detector. Cycle-level analytics (drive / charge / ignition-cycle summaries) and lifetime health tracking are recommended production extensions, not features that already exist in the runtime today.

```mermaid
sequenceDiagram
    participant Transport
    participant EdgeRuntime
    participant InferencePipeline
    participant Storage

    loop Until transport exhausted or SIGTERM
        Transport->>EdgeRuntime: batch(50 rows from transport)
        EdgeRuntime->>EdgeRuntime: Append to rolling buffer
        EdgeRuntime->>InferencePipeline: run_streaming(buffer, new_batch)
        InferencePipeline-->>EdgeRuntime: scored_batch
        EdgeRuntime->>EdgeRuntime: Throttle alerts (cooldown per channel+fault)
        EdgeRuntime->>EdgeRuntime: Write heartbeat every N loop iterations
        EdgeRuntime->>EdgeRuntime: Check model hot-reload (mtime)
    end
    EdgeRuntime->>Storage: write_scored(all_accumulated)
    EdgeRuntime->>Storage: write_alerts(all_alerts)
```

Recommended production layering:

1. Fast streaming layer: short rolling-window detection for overload spikes, voltage sag, protection trips, and thermal excursions.
2. Cycle layer: per-drive / per-charge / per-ignition summaries for repeated stress, nuisance trips, and operating-context scoring.
3. Lifetime layer: long-horizon degradation tracking for aging, drift, and service diagnostics.

## Module Dependency Graph

```mermaid
graph TD
    SCH[schemas/telemetry] --> CFG[config/models]
    SCH --> CAT[config/catalog]
    SCH --> SIM[simulation/generator]
    SCH --> ING[ingestion/normalizer]
    SCH --> MOD[models/anomaly]
    SCH --> INF[inference/pipeline]

    CAT --> SIM
    CFG --> SIM
    CFG --> FE[features/engine]
    CFG --> MOD
    CFG --> INF
    CFG --> STR[storage/writer]

    LOG[utils/logging] --> SIM
    LOG --> ING
    LOG --> FE
    LOG --> MOD
    LOG --> INF
    LOG --> TRN[transport/mock_can]
    LOG --> STR
    LOG --> EDG[edge/runtime]
    LOG --> CLI[cli]

    FE --> INF
    MOD --> INF
    INF --> EDG
    TRN --> EDG
    STR --> EDG
```

**Leaf modules** (no internal dependencies): `schemas/telemetry`, `utils/logging`.

## Vehicle Topology

### eFuse IC Catalog — 17 Families

Each family defines electrical and thermal defaults used by the simulator.

| Family | Nominal A | Max A | R_ds_on (Ω) | R_thermal (°C/W) | τ_thermal (s) |
|--------|-----------|-------|-------------|-------------------|---------------|
| `HS_2A` | 1.5 | 3 | 0.180 | 80 | 8 |
| `HS_5A` | 3.5 | 7 | 0.060 | 55 | 12 |
| `HS_10A` | 6 | 15 | 0.025 | 40 | 15 |
| `HS_15A` | 10 | 20 | 0.012 | 35 | 18 |
| `HS_20A` | 14 | 28 | 0.008 | 30 | 20 |
| `HS_30A` | 20 | 40 | 0.005 | 25 | 22 |
| `HS_50A` | 35 | 65 | 0.003 | 18 | 30 |
| `LS_5A` | 3 | 7 | 0.070 | 60 | 10 |
| `LS_15A` | 10 | 20 | 0.015 | 35 | 15 |

### Example Topology — 4 Zones, 65 Channels

| Zone | Location | Bus | CDD Cycle | Channels | Example Loads |
|------|----------|-----|-----------|----------|---------------|
| Rear | Rear module | CAN | 100ms | 25 | Seating, lighting, infotainment, ADAS, drivetrain |
| Body | Passenger cabin | CAN | 100ms | 15 | Doors, locks, cabin climate, body electronics |
| Front | Front module | CAN | 50ms | 15 | Power supply, HVAC, suspension, auxiliary loads |
| Central | Central gateway | CAN | 50ms | 10 | Power distribution, closures, reserves |

### Channel Factory

The `build_channels()` factory expands compact channel specs into full `ChannelMeta` objects:

1. Look up the eFuse family profile from `EFUSE_CATALOG`
2. Apply catalog defaults (R_ds_on, R_thermal, τ, nominal/max current)
3. Apply per-channel overrides (custom thresholds, load type)
4. Inherit `zone_id` and `source_protocol` from the Zone Controller

## Multi-Rate Protocol Support

| Protocol | Use Case | Rates | Transport Class |
|----------|----------|-------|-----------------|
| **CAN** | Production vehicles | 50–100ms per channel | `CanTransport` |
| **XCP** | Test benches | 10ms (fast) + 50ms (slow) dual-raster | `XcpTransport` |
| **Replay** | Offline analysis | From recorded Parquet/CSV | `ReplayTransport` |

The `FeatureConfig.resolve(sample_interval_s)` method auto-computes `window_size` and `min_periods` from time-domain settings (`window_duration_s`, `min_duration_s`), so the same config works across CAN (100ms) and XCP (10ms) without manual tuning.

## Data Schemas

### Telemetry Record (one row per sample per channel)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `timestamp` | datetime | UTC | Sample time |
| `channel_id` | str | — | eFuse channel identifier |
| `current_a` | float | -1 to 200 A | Measured current |
| `voltage_v` | float | 0 to 60 V | Measured voltage |
| `temperature_c` | float | -40 to 150 °C | Junction temperature |
| `state_on_off` | bool | — | Channel power state |
| `trip_flag` | bool | — | Over-current trip active |
| `overload_flag` | bool | — | Overload condition detected |
| `reset_counter` | int | ≥ 0 | Cumulative reset count |
| `pwm_duty_pct` | float | 0–100 | PWM duty cycle % |
| `device_status` | enum | ok/warning/fault/unknown | Channel health state |

### ChannelMeta — Per-Channel Configuration

| Field | Type | Description |
|-------|------|-------------|
| `channel_id` | str | Unique ID (e.g., `body_interior_light_1`) |
| `efuse_family` | EFuseFamily | IC type from catalog (e.g., `HS_5A`) |
| `nominal_current_a` | float | Normal operating current |
| `max_current_a` | float | Absolute maximum before trip |
| `r_ds_on_ohm` | float | MOSFET on-resistance |
| `r_thermal_kw` | float | Thermal resistance (°C/W) |
| `tau_thermal_s` | float | Thermal time constant (s) |
| `adc_bits` | int | ADC resolution (8–16) |
| `pink_noise_alpha` | float | 1/f^α noise shaping (0=white, 1=pink) |
| `emi_amplitude_a` | float | EMI spike amplitude |
| `inrush_factor` | float | Turn-on current multiplier |
| `inrush_duration_ms` | float | Inrush transient decay time |
| `load_type` | str | `resistive`, `motor`, `inductive`, `ptc` |
| `sample_interval_ms` | float | Per-channel rate (0 = use global) |
| `source_protocol` | SourceProtocol | CAN / XCP / REPLAY |
| `zone_id` | str | Zone Controller assignment |
| `connected_loads` | list[str] | Vehicle loads on this channel |

### Derived Features (appended by FeatureEngine)

| Feature | Computation | Purpose |
|---------|-------------|---------|
| `rolling_rms_current` | √(rolling mean of current²) | Load magnitude regardless of sign |
| `rolling_mean_current` | Rolling window mean | Baseline trend |
| `rolling_max_current` | Rolling window max | Peak detection |
| `rolling_min_current` | Rolling window min | Dropout detection |
| `temperature_slope` | Finite difference over window | Thermal runaway detection |
| `spike_score` | (current − μ) / σ, clipped ≥ 0 | How many σ above normal |
| `trip_frequency` | Rolling sum of trip edges | Repeated protection triggers |
| `recovery_time_s` | Time since last trip release | How fast the channel recovers |
| `degradation_trend` | Least-squares slope of rolling mean | Long-term current increase |
| `missing_rate` | Rolling NaN ratio (pre-ffill) | Packet loss indicator |

### Inference Output

| Field | Type | Description |
|-------|------|-------------|
| `is_anomaly` | bool | Isolation Forest prediction |
| `anomaly_score` | float 0–1 | Higher = more anomalous |
| `predicted_fault` | FaultType enum | Rules classifier output |
| `fault_confidence` | float 0–1 | Classifier certainty |
| `likely_causes` | list[str] | Human-readable explanations |
| `recommended_action` | str | Suggested next step |

## Physics-Based Signal Models

### Thermal Model — First-Order RC

Junction temperature is computed iteratively per sample:

```
T_junction = T_ambient + P · R_thermal · (1 − e^(−t/τ))
P = I² · R_ds_on
```

Where `R_thermal`, `τ` (tau), and `R_ds_on` come from the eFuse family profile. Dynamic power dissipation is recalculated at each sample for realistic thermal response to load changes.

### Load-Specific Inrush Transients

| Load Type | Inrush Multiplier | Duration | Example |
|-----------|------------------|----------|---------|
| Resistive | 1.0× | — | Heaters, lights |
| Motor | 5.0× | 50ms | Power windows, wipers |
| Inductive | 3.0× | 20ms | Solenoids, horn, relays |
| PTC | 2.0× | 500ms | Cold-start thermistors |

### Composite Noise Model

Four noise components are summed per channel:

1. **Pink noise** (1/f^α) — spectral shaping with configurable α
2. **ADC quantization** — LSB = full_range / 2^adc_bits
3. **Thermal noise** — scales with √(T/T_ref)
4. **EMI spikes** — sporadic bursts at configurable amplitude

### eFuse Protection Simulation

Realistic trip → off → cooldown → retry → latch-off cycle:

1. **Detect overcurrent** (~3ms response time)
2. **Trip:** turn off channel, current drops to leakage
3. **Cooldown:** configurable per IC family (0.5–3s)
4. **Retry:** turn channel back on (may re-trip if fault persists)
5. **Latch off:** after max retries exhausted, channel stays off

## Simulated Fault Types

| Fault | What the simulator does | How the classifier detects it |
|-------|------------------------|-------------------------------|
| `overload_spike` | Spike current to 150% of max for ~3s, set trip_flag | spike_score > 4 AND trip_flag |
| `intermittent_overload` | Randomly spike 30% of samples | trip_frequency > 2 AND overload_flag |
| `voltage_sag` | Drop voltage to ~70% of nominal | voltage_v < 11 V |
| `thermal_drift` | Ramp temperature +40°C over window | temperature_slope > 0.3 |
| `noisy_sensor` | Add high-variance Gaussian noise | spike_score > 2.5 without trip/overload |
| `dropped_packet` | Set current/voltage to NaN randomly | missing_rate > 0.1 |
| `gradual_degradation` | Linear increase in current + temperature | degradation_trend > 0.01 |

## Edge Runtime Hardening

| Feature | Description | Config |
|---------|-------------|--------|
| Alert rate-limiting | Suppress duplicate channel+fault alerts within cooldown window | `alert_cooldown_s` (default 10s) |
| Heartbeat | Write `heartbeat.json` every N iterations for external monitoring | `heartbeat_interval` (default 5) |
| Model hot-reload | Detect model file mtime change, reload without restart | `model_hot_reload` (default true) |
| Signal handling | SIGINT/SIGTERM → graceful shutdown (finish current batch) | Always on |
| Error resilience | Track consecutive errors, crash after threshold | `max_consecutive_errors` (default 5) |
| Disk protection | Skip writes when free space drops below threshold | `disk_min_free_mb` (default 100) |
| Iteration metrics | Per-batch stats: inference_ms, rows, alerts, memory RSS | `RuntimeStats` / `IterationStats` |

## Structured Logging

All log records include a correlation `run_id` (format: `YYYYMMDD-HHMMSS-xxxx`) that ties together every log line from a single pipeline run.

| Mode | Flag | Output |
|------|------|--------|
| Pretty (default) | — | `2026-04-04 12:00:00 | module | INFO | [run_id] message` |
| JSON | `--json-log` | `{"ts":"…","level":"INFO","logger":"module","msg":"…","run_id":"…"}` |

JSON mode is designed for log aggregation pipelines (ELK, Datadog, etc.).

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Data contracts | Pydantic 2.x | Typed validation at system boundaries |
| Signal processing | pandas + numpy | Rolling windows, vectorized math |
| Anomaly detection | scikit-learn (Isolation Forest) | Unsupervised, lightweight, no GPU |
| Serialization | Parquet via pyarrow | Columnar, compressed, schema-preserving |
| Configuration | YAML via pyyaml | Human-readable, versionable scenarios |
| CLI | typer + rich | Typed options, formatted output, stderr separation |
| Model persistence | joblib | scikit-learn standard |

## Deployment Topology

| Component | Laptop | Edge (Jetson) |
|-----------|--------|---------------|
| Simulation + training | ✅ | — |
| Feature engine | ✅ | ✅ (smaller buffers) |
| Inference pipeline | ✅ batch | ✅ streaming |
| Edge runtime hardening | ✅ (testing) | ✅ (production) |
| Edge runtime | ✅ testing | ✅ target |
| Storage | ✅ parquet | ✅ parquet |
| Backend sync | — (future) | — (future) |

The trained model artifact (~100KB joblib file) is generated on the laptop and deployed to edge.
