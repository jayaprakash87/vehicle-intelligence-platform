"""Tests for the telemetry schemas."""

from datetime import datetime, timezone

from src.schemas.telemetry import (
    ChannelMeta,
    DerivedFeatures,
    DeviceStatus,
    EFuseFamily,
    EFuseProfile,
    EventLabel,
    FaultInjection,
    FaultType,
    InferenceResult,
    SafetyLevel,
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


# ---------------------------------------------------------------------------
# EFuseProfile — dual ADC, F(i,t), supply voltage, safety level
# ---------------------------------------------------------------------------

def test_efuse_profile_dual_adc_defaults():
    p = EFuseProfile(
        efuse_family=EFuseFamily.HS_10A,
        nominal_current_a=6.0, max_current_a=15.0, fuse_rating_a=10.0,
        r_ds_on_ohm=0.025, r_thermal_kw=40.0, tau_thermal_s=15.0,
    )
    assert p.current_adc_bits == 12
    assert p.voltage_adc_bits == 10


def test_efuse_profile_custom_adc():
    p = EFuseProfile(
        efuse_family=EFuseFamily.HS_2A,
        nominal_current_a=1.5, max_current_a=3.0, fuse_rating_a=2.5,
        r_ds_on_ohm=0.180, r_thermal_kw=80.0, tau_thermal_s=8.0,
        current_adc_bits=16, voltage_adc_bits=12,
    )
    assert p.current_adc_bits == 16
    assert p.voltage_adc_bits == 12


def test_efuse_profile_fit_threshold_default_zero():
    """fit_threshold_a2s=0 means 'auto-derive'; generator resolves it."""
    p = EFuseProfile(
        efuse_family=EFuseFamily.HS_15A,
        nominal_current_a=10.0, max_current_a=20.0, fuse_rating_a=15.0,
        r_ds_on_ohm=0.012, r_thermal_kw=35.0, tau_thermal_s=18.0,
    )
    assert p.fit_threshold_a2s == 0.0
    assert p.short_circuit_threshold_a == 0.0


def test_efuse_profile_explicit_fit_threshold():
    p = EFuseProfile(
        efuse_family=EFuseFamily.HS_30A,
        nominal_current_a=20.0, max_current_a=40.0, fuse_rating_a=30.0,
        r_ds_on_ohm=0.005, r_thermal_kw=25.0, tau_thermal_s=22.0,
        fit_threshold_a2s=9.0, short_circuit_threshold_a=120.0,
    )
    assert p.fit_threshold_a2s == 9.0
    assert p.short_circuit_threshold_a == 120.0


def test_efuse_profile_safety_level():
    p = EFuseProfile(
        efuse_family=EFuseFamily.HS_50A,
        nominal_current_a=35.0, max_current_a=65.0, fuse_rating_a=50.0,
        r_ds_on_ohm=0.003, r_thermal_kw=18.0, tau_thermal_s=30.0,
        safety_level=SafetyLevel.ASIL_B,
    )
    assert p.safety_level == SafetyLevel.ASIL_B


def test_safety_level_enum_values():
    assert SafetyLevel.QM.value == "qm"
    assert SafetyLevel.ASIL_D.value == "asil_d"


# ---------------------------------------------------------------------------
# ChannelMeta — dual ADC + protection fields
# ---------------------------------------------------------------------------

def test_channel_meta_dual_adc():
    ch = ChannelMeta(channel_id="ch_001", current_adc_bits=16, voltage_adc_bits=10)
    assert ch.current_adc_bits == 16
    assert ch.voltage_adc_bits == 10


def test_channel_meta_fit_fields():
    ch = ChannelMeta(
        channel_id="ch_001",
        fit_threshold_a2s=2.25,
        short_circuit_threshold_a=60.0,
    )
    assert ch.fit_threshold_a2s == 2.25
    assert ch.short_circuit_threshold_a == 60.0


def test_channel_meta_defaults_zero_protection():
    """Defaults of 0 mean auto-derive in the generator."""
    ch = ChannelMeta(channel_id="ch_001")
    assert ch.fit_threshold_a2s == 0.0
    assert ch.short_circuit_threshold_a == 0.0
