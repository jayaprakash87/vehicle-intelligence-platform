# Developer Reference

Interface contracts, CLI commands, and config scenarios for the implemented streaming layer. For system architecture, see [01_architecture.md](01_architecture.md). For the next engineering work (cycle and lifetime health), see [06_cycle_and_lifetime_health_plan.md](06_cycle_and_lifetime_health_plan.md).

---

## What Is Built

The streaming fault-intelligence pipeline is complete. 248 tests pass.

| Layer | What | Key files |
|-------|------|-----------|
| Foundation | Schemas, config, eFuse catalog (17 families), 6 YAML configs | `src/schemas/`, `src/config/` |
| Simulation | Physics-based telemetry generation | `src/simulation/generator.py` |
| Signal Processing | Multi-rate normalizer, 10 rolling features | `src/ingestion/`, `src/features/` |
| Intelligence | Isolation Forest + rules-based fault classifier (7 types) | `src/models/`, `src/inference/` |
| Runtime | CAN/XCP/Replay transport, hardened edge loop, storage, CLI | `src/transport/`, `src/edge/`, `src/storage/`, `src/cli.py` |

---

## Interface Contracts

Exact method signatures as implemented. All DataFrame parameters and returns are `pd.DataFrame` unless noted.

### Config

```python
class PlatformConfig(BaseModel):
    simulation: SimulationConfig   # channels, faults, timing, seed, topology
    normalizer: NormalizerConfig   # resample_interval_ms, ffill_tolerance_s, missing_rate_window_s
    features:   FeatureConfig      # window_duration_s=5.0, min_duration_s=1.0
    model:      ModelConfig        # contamination=0.05, n_estimators=100, model_dir
    storage:    StorageConfig      # output_dir="output", format="parquet"
    edge:       EdgeConfig         # batch_size, alert_cooldown_s, heartbeat, hot-reload, …

def load_config(path: str | Path) -> PlatformConfig
def default_config() -> PlatformConfig
```

```python
class FeatureConfig(BaseModel):
    window_duration_s: float = 5.0
    min_duration_s: float = 1.0

    def resolve(self, sample_interval_s: float) -> tuple[int, int]:
        """Convert time-domain settings to (window_size, min_periods)."""
```

```python
class NormalizerConfig(BaseModel):
    resample_interval_ms: float = 0.0     # 0 = no resampling
    ffill_tolerance_s: float = 0.5
    missing_rate_window_s: float = 5.0
```

```python
class EdgeConfig(BaseModel):
    batch_size: int = 50  # rows pulled from transport per loop iteration
    alert_anomaly_threshold: float = 0.6
    flush_interval: int = 20  # flush scored data every N loop iterations
    max_consecutive_errors: int = 5
    alert_cooldown_s: float = 10.0
    heartbeat_interval: int = 5  # write heartbeat every N loop iterations
    disk_min_free_mb: int = 100
    model_hot_reload: bool = True
```

### eFuse Catalog & Topology

```python
# config/catalog.py
EFUSE_CATALOG: dict[EFuseFamily, EFuseProfile]    # 17 IC family profiles

def example_topology() -> tuple[list[ZoneController], list[dict]]:
    """65-channel, 4-zone example configuration."""

def build_channels(
    zones: list[ZoneController],
    channel_specs: list[dict],
) -> list[ChannelMeta]:
    """Expand compact specs into full ChannelMeta using catalog defaults."""
```

### Simulation

```python
class TelemetryGenerator:
    def __init__(self, config: SimulationConfig) -> None
    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame]
    #                       (telemetry_df,     labels_df)
```

Internally uses per-channel vectorized generation with:
- `_rc_thermal()` — first-order RC junction temperature model
- `_composite_noise()` — pink + quantization + thermal + EMI noise
- `_apply_protection()` — trip/cooldown/retry/latch-off cycle
- `_apply_inrush()` — load-type-specific turn-on transients
- `_generate_bus_voltage()` — 13.5V + alternator ripple + drift

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

class XcpTransport(DataFrameTransport):
    """XCP DAQ with dual-raster (fast_raster_ms / slow_raster_ms)."""
    def __init__(self, df, fast_raster_ms=10, slow_raster_ms=50)

class CanTransport(DataFrameTransport):
    """Tags rows with CAN source protocol."""
```

### Ingestion

```python
class Normalizer:
    def __init__(self, config: NormalizerConfig | None = None)
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

### Logging

```python
def configure_logging(
    *, level: int = INFO, json_format: bool = False, run_id: str | None = None,
) -> str:
    """Configure root logger once. Returns the active run_id."""

def set_run_id(run_id: str | None = None) -> str
def get_run_id() -> str
def get_logger(name: str, level: int = INFO) -> logging.Logger
```

---

## CLI Commands

| Command | Behaviour | Key Flags |
|---------|-----------|-----------|
| `simulate` | Generate synthetic data → write telemetry + labels to run-ID dir | `--config`, `--output`, `--format`, `--json-log` |
| `train` | Load telemetry → compute features → train IsolationForest → save model | `--config`, `--data`, `--json-log` |
| `infer` | Load telemetry + model → batch inference → write scored output | `--config`, `--data`, `--output`, `--format`, `--json-log` |
| `edge` | Stream telemetry through EdgeRuntime → emit alerts | `--config`, `--data`, `--output`, `--max-iter`, `--json-log` |
| `pipeline` | Full demo: simulate → train → infer (single command) | `--config`, `--output`, `--format`, `--json-log` |

All commands create output in `<output_dir>/<YYYYMMDD-HHMMSS-xxxx>/` to prevent overwriting previous runs.

---

## Config Scenarios

| File | Channels | Purpose |
|------|----------|---------|
| `configs/default.yaml` | 3 | Mixed faults — quick demo and development |
| `configs/nominal.yaml` | — | Clean baseline — no faults injected |
| `configs/stress_test.yaml` | 1 | All 7 faults on a single channel — classifier validation |
| `configs/example_65ch.yaml` | 65 | Full example topology — 4 zones, realistic loads |
| `configs/xcp_test_bench.yaml` | — | XCP dual-raster DAQ (10ms current + 50ms temperature) |
| `configs/production_can.yaml` | — | Production CAN bus rates (50–100ms) |
