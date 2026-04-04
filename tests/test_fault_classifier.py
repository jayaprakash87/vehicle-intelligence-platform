"""Parametrized tests for all 7 fault types in RulesFaultClassifier."""

import pytest

from src.models.anomaly import RulesFaultClassifier
from src.schemas.telemetry import FaultType


_BASE_ROW = {
    "spike_score": 0.1,
    "trip_flag": False,
    "overload_flag": False,
    "trip_frequency": 0,
    "temperature_slope": 0.01,
    "degradation_trend": 0.0,
    "rolling_rms_current": 5.0,
    "current_a": 5.0,
    "voltage_v": 13.5,
    "missing_rate": 0.0,
}


def _row(**overrides) -> dict:
    r = _BASE_ROW.copy()
    r.update(overrides)
    return r


@pytest.mark.parametrize(
    "overrides, expected_fault",
    [
        # 1. Overload spike — high spike + trip
        ({"spike_score": 5.0, "trip_flag": True}, FaultType.OVERLOAD_SPIKE),
        # 2. Intermittent overload — moderate trip_freq + overload_flag
        ({"trip_frequency": 4, "overload_flag": True}, FaultType.INTERMITTENT_OVERLOAD),
        # 3. Voltage sag — low voltage
        ({"voltage_v": 10.0}, FaultType.VOLTAGE_SAG),
        # 4. Thermal drift — high temperature slope
        ({"temperature_slope": 0.8}, FaultType.THERMAL_DRIFT),
        # 5. Gradual degradation — positive trend
        ({"degradation_trend": 0.03}, FaultType.GRADUAL_DEGRADATION),
        # 6. Noisy sensor — high spike, no trip/overload
        (
            {"spike_score": 3.5, "trip_flag": False, "overload_flag": False},
            FaultType.NOISY_SENSOR,
        ),
        # 7. Dropped packet — high missing rate
        ({"missing_rate": 0.25}, FaultType.DROPPED_PACKET),
        # 8. Nominal — all values normal
        ({}, FaultType.NONE),
    ],
    ids=[
        "overload_spike",
        "intermittent_overload",
        "voltage_sag",
        "thermal_drift",
        "gradual_degradation",
        "noisy_sensor",
        "dropped_packet",
        "nominal",
    ],
)
def test_fault_classification(overrides, expected_fault):
    clf = RulesFaultClassifier()
    row = _row(**overrides)
    fault, confidence, causes = clf.classify(row)
    assert fault == expected_fault, f"Expected {expected_fault}, got {fault}"
    if expected_fault != FaultType.NONE:
        assert confidence > 0
        assert len(causes) > 0
    else:
        assert confidence == 0.0


def test_confidence_bounded():
    """Confidence should always be in [0, 1]."""
    clf = RulesFaultClassifier()
    extreme_rows = [
        _row(spike_score=100.0, trip_flag=True),
        _row(voltage_v=0.0),
        _row(temperature_slope=10.0),
        _row(missing_rate=1.0),
        _row(degradation_trend=1.0),
    ]
    for row in extreme_rows:
        _, conf, _ = clf.classify(row)
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of bounds for {row}"


def test_classify_df():
    """classify_df should return correctly shaped DataFrame."""
    import pandas as pd

    clf = RulesFaultClassifier()
    rows = [
        _row(spike_score=5.0, trip_flag=True),
        _row(voltage_v=10.0),
        _row(),
    ]
    df = pd.DataFrame(rows)
    result = clf.classify_df(df)
    assert list(result.columns) == ["predicted_fault", "fault_confidence", "likely_causes"]
    assert len(result) == 3
    assert result.iloc[0]["predicted_fault"] == FaultType.OVERLOAD_SPIKE.value
    assert result.iloc[2]["predicted_fault"] == FaultType.NONE.value


def test_priority_dropped_packet_over_others():
    """Dropped packet should take priority when missing_rate is high, even with other triggers."""
    clf = RulesFaultClassifier()
    row = _row(missing_rate=0.3, spike_score=5.0, trip_flag=True)
    fault, _, _ = clf.classify(row)
    assert fault == FaultType.DROPPED_PACKET
