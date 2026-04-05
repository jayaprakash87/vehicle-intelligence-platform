"""Configuration models and YAML/JSON loading.

All runtime behavior is driven by config objects so scenarios are
reproducible and parameters are never buried in code.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.schemas.telemetry import ChannelMeta, FaultInjection, PowerState, ZoneController


# ---------------------------------------------------------------------------
# Power state timeline
# ---------------------------------------------------------------------------


class PowerStateEvent(BaseModel):
    """A power-state transition at a given time offset in the scenario.

    Example timeline — ignition cycle with cold crank:
      t=0s  SLEEP   (vehicle parked)
      t=5s  CRANK   (starter engaged)
      t=8s  ACTIVE  (engine running, KL15 on)
      t=55s SLEEP   (ignition off)
    """

    time_s: float = Field(ge=0.0, description="Seconds from scenario start for this transition")
    state: PowerState = Field(description="Target power state at this time")


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
    zones: list[ZoneController] = Field(
        default_factory=list, description="Zone Controllers in the vehicle"
    )
    channels: list[ChannelMeta] = Field(
        default_factory=lambda: [
            ChannelMeta(channel_id="ch_01", load_name="headlamp_left", nominal_current_a=6.0),
            ChannelMeta(channel_id="ch_02", load_name="rear_defroster", nominal_current_a=12.0),
            ChannelMeta(channel_id="ch_03", load_name="seat_heater", nominal_current_a=8.0),
        ]
    )
    # Compact channel definitions — expanded via catalog if present
    channel_specs: list[dict] = Field(
        default_factory=list,
        description="Compact channel specs referencing eFuse catalog. Expanded to channels by build_channels().",
    )
    fault_injections: list[FaultInjection] = Field(default_factory=list)
    power_state_events: list["PowerStateEvent"] = Field(
        default_factory=list,
        description=(
            "Ordered list of power-state transitions. Empty = always ACTIVE. "
            "First entry need not start at t=0 — state before first event is ACTIVE."
        ),
    )
    use_example_topology: bool = Field(
        default=False,
        description="When True, populate channels from the built-in 65-channel example topology.",
    )


# ---------------------------------------------------------------------------
# Feature config — time-based (auto-computes sample counts from interval)
# ---------------------------------------------------------------------------


class FeatureConfig(BaseModel):
    window_duration_s: float = Field(default=5.0, description="Rolling window duration in seconds")
    min_duration_s: float = Field(
        default=1.0, description="Minimum data duration before features are valid"
    )
    # Legacy sample-count fields — used if > 0, else auto-computed from duration
    window_size: int = Field(
        default=0, description="Override: fixed window in samples (0=auto from duration)"
    )
    min_periods: int = Field(
        default=0, description="Override: fixed min_periods (0=auto from duration)"
    )

    def resolve(self, sample_interval_s: float) -> tuple[int, int]:
        """Return (window_size, min_periods) for a given sample interval."""
        if self.window_size > 0:
            w = self.window_size
        else:
            w = max(int(self.window_duration_s / sample_interval_s), 2)
        if self.min_periods > 0:
            mp = self.min_periods
        else:
            mp = max(int(self.min_duration_s / sample_interval_s), 1)
        return w, min(mp, w)


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
    batch_size: int = 50  # rows pulled from transport per loop iteration
    alert_anomaly_threshold: float = 0.6  # minimum score required before emitting an alert
    flush_interval: int = 20  # flush scored data to disk every N loop iterations
    max_consecutive_errors: int = 5  # crash after N consecutive loop errors
    alert_cooldown_s: float = 10.0  # suppress duplicate channel+fault alerts within window
    heartbeat_interval: int = 5  # write heartbeat file every N loop iterations
    disk_min_free_mb: int = 100  # warn + skip writes below this threshold
    model_hot_reload: bool = True  # watch model file mtime and reload on change

    # Cycle tracking
    cycle_tracking_enabled: bool = False  # enable cycle accumulation in the edge loop
    cycle_type: str = "ignition"  # label: ignition, drive, charge
    cycle_boundary_column: str | None = "state_on_off"  # column for open/close detection

    # Lifetime health tracking (requires cycle_tracking_enabled)
    lifetime_tracking_enabled: bool = False  # enable lifetime health updates on cycle close
    lifetime_trend_window: int = 5  # number of recent scores for trend detection
    lifetime_upper_bins: int = 2  # how many top bins count as "stressed"

    # DTC debounce / healing (AUTOSAR Dem counter-based)
    dtc_enabled: bool = True  # apply debounce gate before emitting fault alerts
    dtc_fail_threshold: int = Field(
        default=3,
        ge=1,
        description="Consecutive failing evals required to confirm a DTC (PENDING → CONFIRMED)",
    )
    dtc_heal_threshold: int = Field(
        default=10,
        ge=1,
        description="Consecutive passing evals required to clear a confirmed DTC (HEALING → ABSENT)",
    )


# ---------------------------------------------------------------------------
# MQTT config
# ---------------------------------------------------------------------------


class MqttConfig(BaseModel):
    enabled: bool = False
    broker_host: str = "localhost"
    broker_port: int = 1883
    topic_prefix: str = "vip/alerts"
    client_id: str = ""
    username: str = ""
    password: str = ""
    qos: int = 1
    tls: bool = False


# ---------------------------------------------------------------------------
# Normalizer config
# ---------------------------------------------------------------------------


class NormalizerConfig(BaseModel):
    ffill_tolerance_s: float = Field(
        default=0.5,
        description="Max forward-fill duration in seconds (auto-computes row limit from sample rate)",
    )
    missing_rate_window_s: float = Field(
        default=5.0,
        description="Rolling window for missing-data rate calculation in seconds",
    )
    resample_interval_ms: float = Field(
        default=0.0,
        description="Resample all channels to this interval. 0 = no resampling (keep native rates).",
    )


# ---------------------------------------------------------------------------
# Top-level platform config
# ---------------------------------------------------------------------------


class PlatformConfig(BaseModel):
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    edge: EdgeConfig = Field(default_factory=EdgeConfig)
    normalizer: NormalizerConfig = Field(default_factory=NormalizerConfig)
    mqtt: MqttConfig = Field(default_factory=MqttConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> PlatformConfig:
    """Load a PlatformConfig from a YAML or JSON file.

    After parsing, resolves the topology:
      - If use_example_topology is True, populates channels from example_topology()
      - If channel_specs are present, expands them via the eFuse catalog
      - Otherwise uses the explicit channels list as-is
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = PlatformConfig.model_validate(raw)
    _resolve_topology(cfg)
    return cfg


def default_config() -> PlatformConfig:
    """Return a sensible default config for quick starts."""
    return PlatformConfig()


def _resolve_topology(cfg: PlatformConfig) -> None:
    """Populate simulation channels from topology or channel_specs."""
    from src.config.catalog import build_channels, example_topology

    sim = cfg.simulation

    if sim.use_example_topology:
        zones, specs = example_topology()
        sim.zones = zones
        sim.channels = build_channels(zones, specs)
        return

    if sim.channel_specs:
        sim.channels = build_channels(sim.zones, sim.channel_specs)
        return

    # Otherwise: use explicit channels list as-is
