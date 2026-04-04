"""Canonical telemetry and metadata schemas for the Vehicle Intelligence Platform.

All data flowing through the system conforms to these Pydantic models.
Validation happens at system boundaries (ingestion, inference output).
Internal hot-path code uses dicts/DataFrames for speed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DeviceStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    FAULT = "fault"
    UNKNOWN = "unknown"


class FaultType(str, Enum):
    NONE = "none"
    OVERLOAD_SPIKE = "overload_spike"
    INTERMITTENT_OVERLOAD = "intermittent_overload"
    VOLTAGE_SAG = "voltage_sag"
    THERMAL_DRIFT = "thermal_drift"
    NOISY_SENSOR = "noisy_sensor"
    DROPPED_PACKET = "dropped_packet"
    GRADUAL_DEGRADATION = "gradual_degradation"


# ---------------------------------------------------------------------------
# Core telemetry record — one row per sample per channel
# ---------------------------------------------------------------------------

class TelemetryRecord(BaseModel):
    """Single telemetry sample from one eFuse / channel."""

    timestamp: datetime
    channel_id: str
    current_a: float = Field(description="Measured current in amps")
    voltage_v: float = Field(description="Measured voltage in volts")
    temperature_c: float = Field(description="Junction / board temperature °C")
    state_on_off: bool = Field(description="Channel power state")
    trip_flag: bool = Field(default=False, description="Over-current trip active")
    overload_flag: bool = Field(default=False, description="Overload condition detected")
    reset_counter: int = Field(default=0, ge=0, description="Cumulative reset count")
    pwm_duty_pct: float = Field(default=100.0, ge=0.0, le=100.0, description="PWM duty %")
    device_status: DeviceStatus = DeviceStatus.OK

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


# ---------------------------------------------------------------------------
# Event label — attached to simulated data for supervised evaluation
# ---------------------------------------------------------------------------

class EventLabel(BaseModel):
    """Ground-truth label produced by the simulator for a time window."""

    timestamp: datetime
    channel_id: str
    fault_type: FaultType = FaultType.NONE
    severity: float = Field(default=0.0, ge=0.0, le=1.0, description="0=nominal, 1=critical")
    description: str = ""


# ---------------------------------------------------------------------------
# Channel metadata
# ---------------------------------------------------------------------------

class ChannelMeta(BaseModel):
    """Static metadata describing one eFuse / load channel."""

    channel_id: str
    load_name: str = ""
    nominal_current_a: float = 5.0
    max_current_a: float = 20.0
    nominal_voltage_v: float = 13.5
    fuse_rating_a: float = 15.0

    # Load transient profile
    load_type: str = Field(
        default="resistive",
        description="Load model: resistive | inductive | motor | ptc",
    )
    inrush_factor: float = Field(default=1.0, ge=1.0, description="Turn-on current multiplier vs nominal")
    inrush_duration_ms: float = Field(default=0.0, ge=0.0, description="Inrush transient duration in ms")

    # Thermal model (first-order RC)
    r_ds_on_ohm: float = Field(default=0.010, description="eFuse MOSFET on-resistance Ω")
    r_thermal_kw: float = Field(default=40.0, description="Junction-to-ambient thermal resistance °C/W")
    tau_thermal_s: float = Field(default=15.0, description="Thermal time constant (R_th × C_th) seconds")
    t_ambient_c: float = Field(default=25.0, description="Ambient temperature °C")

    # Noise profile
    adc_bits: int = Field(default=12, ge=8, le=16, description="ADC resolution for quantization noise")
    pink_noise_alpha: float = Field(default=1.0, ge=0.0, le=2.0, description="1/f^α noise exponent, 0=white 1=pink")
    emi_amplitude_a: float = Field(default=0.05, ge=0.0, description="EMI spike amplitude in amps")

    # Protection behavior
    cooldown_s: float = Field(default=1.0, ge=0.0, description="Auto-retry delay after eFuse trip")
    max_retries: int = Field(default=3, ge=0, description="Max auto-retries before latch-off")


class FaultInjection(BaseModel):
    """Specifies a fault to inject during simulation."""

    channel_id: str
    fault_type: FaultType
    start_s: float = Field(description="Seconds from scenario start")
    duration_s: float = Field(default=5.0, description="How long the fault lasts")
    intensity: float = Field(default=0.7, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Derived features record
# ---------------------------------------------------------------------------

class DerivedFeatures(BaseModel):
    """Rolling / derived features computed from raw telemetry."""

    timestamp: datetime
    channel_id: str
    rolling_rms_current: float = 0.0
    rolling_mean_current: float = 0.0
    rolling_max_current: float = 0.0
    rolling_min_current: float = 0.0
    temperature_slope: float = 0.0
    spike_score: float = 0.0
    anomaly_score: float = 0.0
    trip_frequency: float = 0.0
    recovery_time_s: float = 0.0
    degradation_trend: float = 0.0


# ---------------------------------------------------------------------------
# Inference output
# ---------------------------------------------------------------------------

class InferenceResult(BaseModel):
    """Output of the inference pipeline for one evaluation window."""

    timestamp: datetime
    channel_id: str
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    predicted_fault: FaultType = FaultType.NONE
    fault_confidence: float = 0.0
    likely_causes: list[str] = Field(default_factory=list)
    recommended_action: str = ""
