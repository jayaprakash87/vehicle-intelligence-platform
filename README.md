# Vehicle Intelligence Platform

Automotive eFuse telemetry intelligence — physics-based signal synthesis, multi-rate ingestion, anomaly detection, fault classification, and hardened edge inference for production vehicle electrical systems.

## What This Does

Generates realistic eFuse telemetry from a catalog of 9 IC families across a 52-channel, 4-zone vehicle topology. Models first-order RC thermal response, load-specific inrush transients, composite noise (pink + quantization + EMI), and eFuse protection cycles (trip → cooldown → retry → latch-off). Supports CAN production rates and XCP dual-raster test-bench rates. Computes rolling signal features, scores anomalies with Isolation Forest, classifies 7 fault types with interpretable rules, and runs it all in a hardened edge loop with alert throttling, heartbeat monitoring, model hot-reload, and signal-safe shutdown. Replays real measurement data from MDF4, CSV, or Parquet files. Publishes alerts via MQTT. No OEM data or lab hardware required.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Optional: dashboard
pip install -e ".[dashboard]"

# Optional: MDF4 replay support
pip install -e ".[replay]"
```

## Usage

```bash
# Full demo — simulate, train, and infer in one command
vip pipeline

# Individual steps
vip simulate --config configs/default.yaml
vip train --data output/<run_id>/telemetry.parquet
vip infer --data output/<run_id>/telemetry.parquet
vip edge --data output/<run_id>/telemetry.parquet --max-iter 100

# Replay real measurement data
vip replay /path/to/data.mf4 --map-current "I_Ch1_out" --map-voltage "V_Ch1"

# 52-channel sedan topology
vip simulate --config configs/sedan_52ch.yaml

# XCP test-bench dual-raster (10ms + 50ms)
vip simulate --config configs/xcp_test_bench.yaml

# Edge runtime with MQTT alerting
vip edge --data output/<run_id>/telemetry.parquet --mqtt --mqtt-broker 192.168.1.100

# Interactive dashboard
vip dashboard

# Structured JSON logs for aggregation
vip pipeline --json-log
```

All commands accept `--config` for scenario selection, `--output` for output directory, and `--json-log` for structured logging. Each run creates a timestamped subdirectory (`<output>/<YYYYMMDD-HHMMSS-xxxx>/`) so results are never overwritten.

## Quickstart

```bash
# Run the full pipeline in one command
vip pipeline

# Or run the Python quickstart example
python examples/quickstart.py

# View results in the dashboard
vip dashboard --data output/quickstart/
```

## Tests

```bash
pytest tests/ -v          # 246 tests across 20 test files
```

## Project Structure

```
src/
├── cli.py                    # Typer CLI — 8 commands (simulate, replay, train, infer, edge, pipeline, dashboard)
├── schemas/telemetry.py      # Pydantic models: TelemetryRecord, ChannelMeta, ProtectionEvent, EFuseFamily, …
├── config/
│   ├── models.py             # Typed config hierarchy + YAML loader
│   └── catalog.py            # eFuse IC catalog (9 families), vehicle topology factory, channel builder
├── simulation/generator.py   # Physics-based telemetry: RC thermal, inrush, composite noise, protection sim
├── transport/
│   ├── mock_can.py           # TransportBase + DataFrameTransport, XcpTransport, CanTransport, ReplayTransport
│   └── alert_sinks.py        # AlertSinkBase + LogAlertSink + MqttAlertSink
├── ingestion/
│   ├── normalizer.py         # Multi-rate resampling, time-based ffill, missing-rate tracking
│   └── reader.py             # MeasurementReader — MDF4/CSV/Parquet ingest with column mapping
├── features/engine.py        # Rolling feature computation (10+ derived signals) with time-based windows
├── models/
│   ├── anomaly.py            # Isolation Forest + rules-based fault classifier (7 fault types)
│   └── evaluation.py         # Precision/recall/F1, per-fault metrics, time-to-detect
├── inference/pipeline.py     # Feature → score → classify orchestration (batch + streaming)
├── edge/runtime.py           # Hardened loop: alert cooldown, heartbeat, hot-reload, signal handling
├── dashboard/app.py          # Streamlit dashboard: signals, anomalies, faults, protection events, summary (+ disk mode)
├── storage/writer.py         # Parquet/CSV/JSON output
└── utils/logging.py          # Structured logging: JSON + pretty formatters, correlation IDs (run_id)

configs/
├── default.yaml              # 3-channel mixed-fault demo
├── nominal.yaml              # Clean baseline (no faults)
├── stress_test.yaml          # All 7 faults on 1 channel
├── sedan_52ch.yaml           # Full 4-zone, 52-channel sedan topology
├── xcp_test_bench.yaml       # XCP dual-raster (10ms current + 50ms temperature)
└── production_can.yaml       # Production CAN bus rates (50–100ms)

examples/
└── quickstart.py             # End-to-end demo: generate → train → infer → evaluate → save

tests/                        # 20 test files, 246 tests
docs/                         # Architecture, implementation plan, design decisions, product strategy
```

## Documentation

- [Architecture](docs/01_architecture.md) — system design, data flow, module graph, schemas, vehicle topology
- [Implementation Plan](docs/02_implementation_plan.md) — build phases, module interfaces, test inventory
- [Design Decisions](docs/03_design_decisions.md) — ADRs with rationale and trade-offs
- [Product Strategy](docs/04_product_strategy.md) — vision, market positioning, go-to-market, roadmap
- [Changelog](CHANGELOG.md) — release history
