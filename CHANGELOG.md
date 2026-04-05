# Changelog

All notable changes to the Vehicle Intelligence Platform are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] — 2026-04-05

OEM gap analysis implementation — 8 production-realism gaps closed across physics modelling, fault coverage, and power-state management.

### Added

**Gap 1 — ISENSE sensing-chain accuracy**
- `k_ilis`, `k_ilis_tempco_ppm_c`, `r_ilis_ohm`, `r_ilis_tolerance` fields on `ChannelMeta`
- `_apply_isense_chain()` in generator: frozen per-unit δ_r + δ_k scatter with dynamic α_k temperature drift
- Models PROFET+2/VIPower ILIS output as read by the CDD ADC

**Gap 2 — Open-load detection**
- `FaultType.OPEN_LOAD` fault case with near-zero leakage current and gate held ON
- `ProtectionEvent.OPEN_LOAD_DIAG` fires after configurable `ol_blank_time_ms` blank time
- `ol_blank_time_ms`, `ol_threshold_a` fields on `ChannelMeta`
- Classifier rule and pipeline action text

**Gap 3 — DTC debounce and healing**
- `DTCDebouncer` state machine: INACTIVE → PENDING → CONFIRMED → HEALING with configurable confirm/heal thresholds
- `dtc_enabled` flag on `ChannelMeta` to opt channels in/out of DTC logic

**Gap 4 — Rds,on temperature coefficient**
- `rds_on_tempco_exp` field on `ChannelMeta` (power-law exponent, default 2.3 for PROFET+2)
- `_rc_thermal()` updated: `R_ds,on(T) = R_ds,on(25°C) × (T_K/300)^n` positive thermal feedback loop
- Junction temperature capped at `thermal_shutdown_c` before exponentiation to prevent overflow

**Gap 5 — Abnormal bus voltage**
- `FaultType.JUMP_START`, `LOAD_DUMP`, `COLD_CRANK` with physically grounded waveforms
- `ProtectionEvent.OVER_VOLTAGE` for bus events exceeding 16 V
- ISO 16750-2 load-dump model: fast spike to ~40 V with exponential decay
- `rolling_min_voltage`, `rolling_max_voltage` features in feature engine
- Classifier rules ordered before voltage-sag catch-all

**Gap 6 — Wire harness resistance and connector aging**
- `harness_r_ohm`, `connector_r_ohm` fields on `ChannelMeta`
- `FaultType.CONNECTOR_AGING`: exponential-squared fretting-corrosion model (`R_c(t) = R_c0 × (1 + k × t²)`)
- `rolling_voltage_drop` feature
- Generator uses `ch.harness_r_ohm + ch.connector_r_ohm` for all resistive drop calculations

**Gap 7 — Multi-channel die thermal coupling**
- `die_id`, `thermal_coupling_coeff` fields on `ChannelMeta`
- `FaultType.THERMAL_COUPLING`
- `_apply_die_thermal_coupling()` post-processing: channels sharing a `die_id` exchange `k × ΔT_neighbour` heat
- Classifier rule (gentle temp slope, no trip, current within 20% of nominal)

**Gap 8 — Sleep/wake power states**
- `PowerState` enum: `SLEEP`, `ACTIVE`, `ACCESSORY`, `CRANK`
- `FaultType.WAKE_TRANSIENT`
- `sleep_quiescent_ua`, `wake_inrush_factor`, `wake_inrush_duration_ms` fields on `ChannelMeta`
- `PowerStateEvent` model and `power_state_events` list on `SimulationConfig`
- `_build_power_state_array()` using Python list (not numpy) to preserve str-Enum identity in Python 3.11
- Power-state gating loop in `_generate_channel`: KL30/KL15/KLR/KL50 × state matrix
- ALWAYS_ON quiescent dark current during SLEEP; KL15 off during SLEEP/CRANK; KL50 on only during CRANK
- Wake inrush on SLEEP→ACTIVE transition with configurable ramp and duration
- Classifier rule and pipeline action text

### Fixed
- `_composite_noise(n=1)`: `np.std` of a 1-sample array is 0, causing division by `1e-12` → ~10¹¹ A spike → spurious thermal shutdown. Guard added: skip pink-noise normalization when `std < 1e-10`
- Removed hardcoded `harness_r = 0.020` in generator; now reads from `ChannelMeta` fields
- `SimulationConfig.power_state_events` type annotation changed from `list["PowerStateEvent"]` (unresolved forward ref) to `list[PowerStateEvent]`

### Tests
- 84 new tests added across all 8 gaps
- Total: **330 tests** (up from 246)

---

## [0.1.0] — 2026-04-04

First tagged release — feature-complete MVP with production hardening.

### Added
- Physics-based eFuse telemetry generator (RC thermal, F(i,t), composite noise, protection cycles)
- eFuse IC catalog (17 IC families — Infineon + STMicroelectronics, 2A–100A + CUSTOM)
- 65-channel, 4-zone example vehicle topology
- Multi-rate ingestion with CAN production and XCP dual-raster support
- Normalizer with resampling to a common time grid
- Rolling feature engine (RMS current, spike score, temperature slope, trip frequency, degradation trend)
- Isolation Forest anomaly detection with 7-feature input
- Rule-based fault classifier for 7 fault types (overload spike, thermal drift, voltage sag, etc.)
- ProtectionEvent-aware classification (SCP, I2T, thermal shutdown, latch-off)
- Hardened edge runtime (signal handling, model hot-reload, rate-limiting, heartbeat, disk guard)
- MQTT alert publishing with configurable broker, topic, QoS, and TLS
- Real data replay from MDF4, CSV, and Parquet measurement files
- Streamlit dashboard with signals, anomalies, faults, protection events, and summary tabs
- Dashboard disk-loading mode (`vip dashboard --data output/<run_id>/`)
- Typer CLI with 8 commands (simulate, replay, train, infer, edge, pipeline, dashboard)
- Structured JSON logging with correlation IDs (run_id)
- Evaluation module for precision/recall/F1 against ground-truth fault windows
- Quickstart example script (`examples/quickstart.py`)
- GitHub Actions CI (lint + test on push/PR)
- Pre-commit hooks (ruff lint/format + fast pytest)
- Quality-gate regression tests (overload spike F1, thermal/voltage smoke, nominal FP rate)
- 6 YAML scenario configs (default, nominal, stress test, example 65ch, XCP test bench, production CAN)
- 246 tests across 20 test files
