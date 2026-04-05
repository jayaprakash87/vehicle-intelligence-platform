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


# ---------------------------------------------------------------------------
# DTCDebouncer tests
# ---------------------------------------------------------------------------


from src.inference.dtc import DTCDebouncer, DTCStatus  # noqa: E402


def test_dtc_pending_before_confirmed():
    """Fault must be seen fail_threshold times before becoming CONFIRMED."""
    dtc = DTCDebouncer(fail_threshold=3, heal_threshold=5)
    s1 = dtc.update("ch_01", FaultType.OVERLOAD_SPIKE, True)
    s2 = dtc.update("ch_01", FaultType.OVERLOAD_SPIKE, True)
    assert s1 == DTCStatus.PENDING
    assert s2 == DTCStatus.PENDING
    assert not dtc.is_confirmed("ch_01", FaultType.OVERLOAD_SPIKE)
    s3 = dtc.update("ch_01", FaultType.OVERLOAD_SPIKE, True)
    assert s3 == DTCStatus.CONFIRMED
    assert dtc.is_confirmed("ch_01", FaultType.OVERLOAD_SPIKE)


def test_dtc_transient_never_confirms():
    """A single failing eval followed by a pass should silently reset to ABSENT."""
    dtc = DTCDebouncer(fail_threshold=3, heal_threshold=5)
    dtc.update("ch_01", FaultType.VOLTAGE_SAG, True)   # PENDING fail_count=1
    s = dtc.update("ch_01", FaultType.VOLTAGE_SAG, False)  # pass → ABSENT
    assert s == DTCStatus.ABSENT
    assert not dtc.is_confirmed("ch_01", FaultType.VOLTAGE_SAG)


def test_dtc_healing_requires_consecutive_passes():
    """CONFIRMED fault must pass heal_threshold consecutive times before clearing."""
    dtc = DTCDebouncer(fail_threshold=2, heal_threshold=4)
    # Confirm fault
    dtc.update("ch_01", FaultType.THERMAL_DRIFT, True)
    dtc.update("ch_01", FaultType.THERMAL_DRIFT, True)
    assert dtc.is_confirmed("ch_01", FaultType.THERMAL_DRIFT)
    # Partial heal — not yet cleared
    dtc.update("ch_01", FaultType.THERMAL_DRIFT, False)  # HEALING heal_count=1
    dtc.update("ch_01", FaultType.THERMAL_DRIFT, False)  # heal_count=2
    dtc.update("ch_01", FaultType.THERMAL_DRIFT, False)  # heal_count=3
    assert dtc.status("ch_01", FaultType.THERMAL_DRIFT) == DTCStatus.HEALING
    # Final pass clears it
    dtc.update("ch_01", FaultType.THERMAL_DRIFT, False)  # heal_count=4 → ABSENT
    assert dtc.status("ch_01", FaultType.THERMAL_DRIFT) == DTCStatus.ABSENT


def test_dtc_fail_during_healing_reconfirms():
    """A fail during HEALING should push status back to CONFIRMED."""
    dtc = DTCDebouncer(fail_threshold=2, heal_threshold=5)
    dtc.update("ch_01", FaultType.OPEN_LOAD, True)
    dtc.update("ch_01", FaultType.OPEN_LOAD, True)  # CONFIRMED
    dtc.update("ch_01", FaultType.OPEN_LOAD, False)  # HEALING
    dtc.update("ch_01", FaultType.OPEN_LOAD, False)  # HEALING
    s = dtc.update("ch_01", FaultType.OPEN_LOAD, True)  # fault returned
    assert s == DTCStatus.CONFIRMED


def test_dtc_channels_independent():
    """DTC state for different channels must not share counters."""
    dtc = DTCDebouncer(fail_threshold=3, heal_threshold=5)
    dtc.update("ch_01", FaultType.VOLTAGE_SAG, True)
    dtc.update("ch_01", FaultType.VOLTAGE_SAG, True)
    dtc.update("ch_01", FaultType.VOLTAGE_SAG, True)  # ch_01 CONFIRMED
    assert dtc.is_confirmed("ch_01", FaultType.VOLTAGE_SAG)
    assert not dtc.is_confirmed("ch_02", FaultType.VOLTAGE_SAG)  # ch_02 untouched


def test_dtc_reset_channel():
    """reset_channel should clear all DTC records for that channel only."""
    dtc = DTCDebouncer(fail_threshold=2, heal_threshold=5)
    dtc.update("ch_01", FaultType.OVERLOAD_SPIKE, True)
    dtc.update("ch_01", FaultType.OVERLOAD_SPIKE, True)  # ch_01 CONFIRMED
    dtc.update("ch_02", FaultType.VOLTAGE_SAG, True)
    dtc.update("ch_02", FaultType.VOLTAGE_SAG, True)  # ch_02 CONFIRMED
    dtc.reset_channel("ch_01")
    assert dtc.status("ch_01", FaultType.OVERLOAD_SPIKE) == DTCStatus.ABSENT
    assert dtc.is_confirmed("ch_02", FaultType.VOLTAGE_SAG)  # unaffected


def test_dtc_snapshot_excludes_absent():
    """Snapshot should only include non-ABSENT records."""
    dtc = DTCDebouncer(fail_threshold=2, heal_threshold=5)
    dtc.update("ch_01", FaultType.OVERLOAD_SPIKE, True)  # PENDING
    snap = dtc.snapshot()
    assert len(snap) == 1
    key = list(snap.keys())[0]
    assert snap[key]["status"] == DTCStatus.PENDING.value
    dtc.update("ch_01", FaultType.OVERLOAD_SPIKE, False)  # healed → ABSENT
    assert dtc.snapshot() == {}
