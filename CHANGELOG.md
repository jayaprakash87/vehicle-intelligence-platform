# Changelog

All notable changes to the Vehicle Intelligence Platform are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-04-04

First tagged release — feature-complete MVP with production hardening.

### Added
- Physics-based eFuse telemetry generator (RC thermal, F(i,t), composite noise, protection cycles)
- eFuse IC catalog (9 families — BTS7006, VN7140, TLE6240, etc.) with real part numbers
- 52-channel, 4-zone sedan vehicle topology
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
- 6 YAML scenario configs (default, nominal, stress test, sedan 52ch, XCP test bench, production CAN)
- 246 tests across 20 test files
