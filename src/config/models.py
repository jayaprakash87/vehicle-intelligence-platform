"""Configuration models and YAML/JSON loading.

All runtime behavior is driven by config objects so scenarios are
reproducible and parameters are never buried in code.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.schemas.telemetry import ChannelMeta, FaultInjection


# ---------------------------------------------------------------------------
# Simulation config
# ---------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    scenario_id: str = "default"
    name: str = "Default Scenario"
    description: str = ""
    duration_s: float = 60.0
    sample_interval_ms: float = 100.0
    seed: int = 42
    channels: list[ChannelMeta] = Field(default_factory=lambda: [
        ChannelMeta(channel_id="ch_01", load_name="headlamp_left", nominal_current_a=6.0),
        ChannelMeta(channel_id="ch_02", load_name="rear_defroster", nominal_current_a=12.0),
        ChannelMeta(channel_id="ch_03", load_name="seat_heater", nominal_current_a=8.0),
    ])
    fault_injections: list[FaultInjection] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature config
# ---------------------------------------------------------------------------

class FeatureConfig(BaseModel):
    window_size: int = 50
    min_periods: int = 10


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    anomaly_contamination: float | str = Field(default="auto")
    anomaly_n_estimators: int = 100
    model_dir: str = "artifacts/models"


# ---------------------------------------------------------------------------
# Storage config
# ---------------------------------------------------------------------------

class StorageConfig(BaseModel):
    output_dir: str = "output"
    format: str = "parquet"  # "parquet" | "csv" | "json"


# ---------------------------------------------------------------------------
# Edge config
# ---------------------------------------------------------------------------

class EdgeConfig(BaseModel):
    batch_size: int = 50
    alert_anomaly_threshold: float = 0.6
    flush_interval: int = 20  # flush scored data to disk every N batches
    max_consecutive_errors: int = 5  # crash after N consecutive loop errors
    alert_cooldown_s: float = 10.0  # suppress duplicate channel+fault alerts within window
    heartbeat_interval: int = 5  # write heartbeat file every N iterations
    disk_min_free_mb: int = 100  # warn + skip writes below this threshold
    model_hot_reload: bool = True  # watch model file mtime and reload on change


# ---------------------------------------------------------------------------
# Top-level platform config
# ---------------------------------------------------------------------------

class PlatformConfig(BaseModel):
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    edge: EdgeConfig = Field(default_factory=EdgeConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> PlatformConfig:
    """Load a PlatformConfig from a YAML or JSON file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PlatformConfig.model_validate(raw)


def default_config() -> PlatformConfig:
    """Return a sensible default config for quick starts."""
    return PlatformConfig()
