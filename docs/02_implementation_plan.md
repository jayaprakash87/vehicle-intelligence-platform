# Implementation Plan

This document defines the build order, exit criteria, and interface contracts between modules. For system architecture and data flow diagrams, see [01_architecture.md](01_architecture.md).

---

## Build Phases

### Phase 1 — Foundation (schemas + config + simulation)

Build the data contract and synthetic data generator. Nothing downstream can be developed or tested without data.

| File | Responsibility |
|------|----------------|
| `src/schemas/telemetry.py` | Pydantic models for all system boundaries |
| `src/config/models.py` | Typed config hierarchy + YAML loader |
| `src/simulation/generator.py` | Per-channel signal synthesis with fault injection |
| `src/utils/logging.py` | Logger factory |
| `configs/default.yaml` | Mixed-fault demo (3 channels, 4 faults) |
| `configs/nominal.yaml` | Clean baseline |
| `configs/stress_test.yaml` | All 7 faults on 1 channel |

**Exit criteria:** `TelemetryGenerator.generate()` returns well-shaped DataFrames. Fault windows are visible in raw data. YAML configs load without error.

### Phase 2 — Signal Processing (ingestion + features)

Raw data → cleaned → enriched with rolling features.

| File | Responsibility |
|------|----------------|
| `src/ingestion/normalizer.py` | NaN tracking → forward-fill → clip → type coercion |
| `src/features/engine.py` | Rolling window feature computation (10 derived columns) |

**Exit criteria:** Feature columns are present and physically plausible — RMS ≥ 0, spike_score ≥ 0, temperature_slope changes sign around thermal fault windows.

### Phase 3 — Intelligence (models + inference)

Train anomaly detector, build rules classifier, orchestrate scoring.

| File | Responsibility |
|------|----------------|
| `src/models/anomaly.py` | IsolationForest wrapper + threshold-based fault classifier |
| `src/inference/pipeline.py` | Features → anomaly score → fault classification → InferenceResult |

**Exit criteria:** Batch scoring detects anomalies in fault-injected windows. Rules classifier returns correct fault types for known patterns. Model saves and reloads successfully.

### Phase 4 — Runtime (transport + edge + storage + CLI)

Wire into a runnable system with persistence and both operating modes.

| File | Responsibility |
|------|----------------|
| `src/transport/mock_can.py` | ABC + DataFrame/file transport backends |
| `src/edge/runtime.py` | Buffered streaming loop with alert emission |
| `src/storage/writer.py` | Parquet/CSV/JSON output + alert persistence |
| `src/cli.py` | 5 Typer commands |

**Exit criteria:** `python -m src.cli pipeline` runs end-to-end, writes parquet + alerts JSON. `edge` command streams and emits alerts for anomalies.

### Phase 5 — Validation (tests)

| File | Covers |
|------|--------|
| `tests/test_schemas.py` | Model construction, defaults, validation, serialization |
| `tests/test_simulation.py` | Output shape, fault presence, seed reproducibility, multi-channel |
| `tests/test_features.py` | Feature columns present, values plausible |
| `tests/test_models.py` | Train/score/save/load, known-fault classification |
| `tests/test_pipeline.py` | Full integration: simulate → normalize → features → train → infer |

**Exit criteria:** 15/15 tests pass. Integration test detects injected faults.

---

## Interface Contracts

Exact method signatures as implemented. All DataFrame parameters and returns are `pd.DataFrame` unless noted.

### Config

```python
class PlatformConfig(BaseModel):
    simulation: SimulationConfig   # channels, faults, timing, seed
    features:   FeatureConfig      # window_size=50, min_periods=10
    model:      ModelConfig        # contamination=0.05, n_estimators=100, model_dir
    storage:    StorageConfig      # output_dir="output", format="parquet"
    edge:       EdgeConfig         # batch_size=50, alert_anomaly_threshold=0.6

def load_config(path: str | Path) -> PlatformConfig
def default_config() -> PlatformConfig
```

### Simulation

```python
class TelemetryGenerator:
    def __init__(self, config: SimulationConfig) -> None
    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame]
    #                       (telemetry_df,     labels_df)
```

### Transport

```python
class TransportBase(ABC):
    def stream(self) -> Iterator[dict]
    def batch(self, size: int) -> list[dict]

class DataFrameTransport(TransportBase):
    def __init__(self, df: pd.DataFrame, realtime: bool = False, speed: float = 1.0)
    @property
    def exhausted(self) -> bool
    def reset(self) -> None

class ReplayTransport(DataFrameTransport):
    def __init__(self, path: str, realtime: bool = False, speed: float = 1.0)
```

### Ingestion

```python
class Normalizer:
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame
    def validate_record(self, row: dict) -> TelemetryRecord | None
```

### Features

```python
class FeatureEngine:
    def __init__(self, config: FeatureConfig | None = None)
    def compute(self, df: pd.DataFrame) -> pd.DataFrame
```

### Models

```python
class AnomalyDetector:
    def __init__(self, config: ModelConfig | None = None)
    def train(self, df: pd.DataFrame) -> None
    def score(self, df: pd.DataFrame) -> pd.Series     # float 0–1, higher = more anomalous
    def predict(self, df: pd.DataFrame) -> pd.Series    # bool
    def save(self, path: str | Path | None = None) -> Path
    def load(self, path: str | Path | None = None) -> None

class RulesFaultClassifier:
    def classify(self, row: pd.Series | dict) -> tuple[FaultType, float, list[str]]
    def classify_df(self, df: pd.DataFrame) -> pd.DataFrame
```

### Inference

```python
class InferencePipeline:
    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        model_config: ModelConfig | None = None,
        anomaly_model: AnomalyDetector | None = None,
    )
    def run_batch(self, df: pd.DataFrame) -> pd.DataFrame
    def run_streaming(self, buffer: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame
    def to_inference_results(self, df: pd.DataFrame) -> list[InferenceResult]
```

### Edge

```python
class EdgeRuntime:
    def __init__(
        self,
        transport: TransportBase,
        pipeline: InferencePipeline,
        edge_config: EdgeConfig | None = None,
        writer: StorageWriter | None = None,
    )
    def run(self, max_iterations: int | None = None) -> list[dict]
    def stop(self) -> None
```

### Storage

```python
class StorageWriter:
    def __init__(self, config: StorageConfig | None = None)
    def write_telemetry(self, df: pd.DataFrame, name: str = "telemetry") -> Path
    def write_features(self, df: pd.DataFrame, name: str = "features") -> Path
    def write_labels(self, df: pd.DataFrame, name: str = "labels") -> Path
    def write_scored(self, df: pd.DataFrame, name: str = "scored") -> Path
    def write_alerts(self, alerts: list[dict], name: str = "alerts") -> Path
```

---

## CLI Commands

| Command | Behaviour |
|---------|-----------|
| `simulate` | Generate synthetic data → write telemetry + labels to output/ |
| `train` | Load telemetry → compute features → train IsolationForest → save model |
| `infer` | Load telemetry + model → batch inference → write scored output |
| `edge` | Stream telemetry through EdgeRuntime → emit alerts |
| `pipeline` | Full demo: simulate → train → infer (single command) |

All commands accept `--config`, `--output`, `--format` flags.
