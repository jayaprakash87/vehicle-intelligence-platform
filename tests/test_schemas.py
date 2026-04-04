"""Tests for the telemetry schemas."""

from datetime import datetime, timezone

from src.schemas.telemetry import (
    ChannelMeta,
    DerivedFeatures,
    DeviceStatus,
    EventLabel,
    FaultInjection,
    FaultType,
    InferenceResult,
    TelemetryRecord,
)


def test_telemetry_record_defaults():
    rec = TelemetryRecord(
        timestamp=datetime.now(tz=timezone.utc),
        channel_id="ch_01",
        current_a=5.5,
        voltage_v=13.4,
        temperature_c=28.0,
        state_on_off=True,
    )
    assert rec.trip_flag is False
    assert rec.overload_flag is False
    assert rec.reset_counter == 0
    assert rec.pwm_duty_pct == 100.0
    assert rec.device_status == DeviceStatus.OK


def test_fault_injection_schema():
    fi = FaultInjection(
        channel_id="ch_01",
        fault_type=FaultType.OVERLOAD_SPIKE,
        start_s=10.0,
        duration_s=5.0,
        intensity=0.8,
    )
    assert fi.intensity == 0.8


def test_event_label_schema():
    label = EventLabel(
        timestamp=datetime.now(tz=timezone.utc),
        channel_id="ch_01",
        fault_type=FaultType.THERMAL_DRIFT,
        severity=0.6,
    )
    assert label.fault_type == FaultType.THERMAL_DRIFT


def test_inference_result_defaults():
    result = InferenceResult(
        timestamp=datetime.now(tz=timezone.utc),
        channel_id="ch_01",
    )
    assert result.is_anomaly is False
    assert result.predicted_fault == FaultType.NONE
    assert result.likely_causes == []
