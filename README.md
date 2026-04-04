# Vehicle Intelligence Platform

Automotive eFuse telemetry intelligence — physics-based signal synthesis, multi-rate ingestion, anomaly detection, fault classification, and hardened edge inference for production vehicle electrical systems.

## What This Does

Generates realistic eFuse telemetry from a catalog of 9 IC families across a 52-channel, 4-zone vehicle topology. Models first-order RC thermal response, load-specific inrush transients, composite noise (pink + quantization + EMI), and eFuse protection cycles (trip → cooldown → retry → latch-off). Supports CAN production rates and XCP dual-raster test-bench rates. Computes rolling signal features, scores anomalies with Isolation Forest, classifies 7 fault types with interpretable rules, and runs it all in a hardened edge loop with alert throttling, heartbeat monitoring, model hot-reload, and signal-safe shutdown. No OEM data or lab hardware required.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Full demo — simulate, train, and infer in one command
python -m src.cli pipeline

# Individual steps
python -m src.cli simulate --config configs/default.yaml
python -m src.cli train --data output/telemetry.parquet
python -m src.cli infer --data output/telemetry.parquet
python -m src.cli edge --data output/telemetry.parquet --max-iter 100

# 52-channel sedan topology
python -m src.cli simulate --config configs/sedan_52ch.yaml

# XCP test-bench dual-raster (10ms + 50ms)
python -m src.cli simulate --config configs/xcp_test_bench.yaml

# Structured JSON logs for aggregation
python -m src.cli pipeline --json-log
```

All commands accept `--config` for scenario selection, `--output` for output directory, and `--json-log` for structured logging. Each run creates a timestamped subdirectory (`<output>/<YYYYMMDD-HHMMSS-xxxx>/`) so results are never overwritten.

## Tests

```bash
pytest tests/ -v          # 184 tests across 17 test files
```

## Project Structure

```
src/
├── cli.py                  # Typer CLI — 5 commands with error handling & run-ID isolation
├── schemas/telemetry.py    # Pydantic models: TelemetryRecord, ChannelMeta, ProtectionEvent, EFuseFamily, …
├── config/
│   ├── models.py           # Typed config hierarchy + YAML loader
│   └── catalog.py          # eFuse IC catalog (9 families), vehicle topology factory, channel builder
├── simulation/generator.py # Physics-based telemetry: RC thermal, inrush, composite noise, protection sim
├── transport/mock_can.py   # TransportBase + DataFrameTransport, XcpTransport, CanTransport, ReplayTransport
├── ingestion/normalizer.py # Multi-rate resampling, time-based ffill, missing-rate tracking
├── features/engine.py      # Rolling feature computation (10 derived signals) with time-based windows
├── models/anomaly.py       # Isolation Forest + threshold rules classifier (7 fault types)
├── inference/pipeline.py   # Feature → score → classify orchestration (batch + streaming)
├── edge/runtime.py         # Hardened loop: alert cooldown, heartbeat, hot-reload, signal handling
├── storage/writer.py       # Parquet/CSV/JSON output
└── utils/logging.py        # Structured logging: JSON + pretty formatters, correlation IDs (run_id)

configs/
├── default.yaml            # 3-channel mixed-fault demo
├── nominal.yaml            # Clean baseline (no faults)
├── stress_test.yaml        # All 7 faults on 1 channel
├── sedan_52ch.yaml         # Full 4-zone, 52-channel sedan topology
├── xcp_test_bench.yaml     # XCP dual-raster (10ms current + 50ms temperature)
└── production_can.yaml     # Production CAN bus rates (50–100ms)

tests/                      # 17 test files, 184 tests
docs/                       # Architecture, implementation plan, design decisions
```

## Documentation

- [Architecture](docs/01_architecture.md) — system design, data flow, module graph, schemas, vehicle topology
- [Implementation Plan](docs/02_implementation_plan.md) — build phases, module interfaces, test inventory
- [Design Decisions](docs/03_design_decisions.md) — ADRs with rationale and trade-offs
- [Product Strategy](docs/04_product_strategy.md) — vision, market positioning, go-to-market, roadmap
