# Vehicle Intelligence Platform

Automotive eFuse telemetry intelligence — synthetic data generation, signal processing, anomaly detection, and edge inference.

## What This Does

Generates realistic eFuse telemetry with configurable fault injection, computes rolling signal features, scores anomalies with Isolation Forest, classifies faults with interpretable rules, and runs it all in a lightweight edge loop. No OEM data or lab hardware required.

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
python -m src.cli edge --data output/telemetry.parquet
```

All commands accept `--config` for scenario selection and `--output` for output directory.

## Tests

```bash
pytest tests/ -v
```

## Project Structure

```
src/
├── cli.py                  # Typer CLI entry point
├── schemas/telemetry.py    # Pydantic data models
├── config/models.py        # Typed config + YAML loader
├── simulation/generator.py # Synthetic telemetry with fault injection
├── transport/mock_can.py   # Stream/batch abstraction
├── ingestion/normalizer.py # Clean, validate, fill gaps
├── features/engine.py      # Rolling feature computation
├── models/anomaly.py       # Isolation Forest + rules classifier
├── inference/pipeline.py   # Feature → score → classify orchestration
├── edge/runtime.py         # Buffered inference loop with alerts
├── storage/writer.py       # Parquet/CSV/JSON output
└── utils/logging.py        # Logger factory
configs/                    # YAML scenario definitions
tests/                      # pytest suite
docs/                       # Architecture, implementation plan, design decisions
```

## Documentation

- [Architecture](docs/01_architecture.md) — system design, data flow, module graph, schemas
- [Implementation Plan](docs/02_implementation_plan.md) — build phases, module interfaces
- [Design Decisions](docs/03_design_decisions.md) — key choices with rationale and trade-offs
