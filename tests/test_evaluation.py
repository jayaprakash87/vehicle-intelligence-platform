"""Tests for src.models.evaluation — precision/recall/F1, per-fault metrics, time-to-detect."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.models.evaluation import _build_fault_windows, _match_fault_window, evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _ts(offset_s: float) -> datetime:
    return _T0 + timedelta(seconds=offset_s)


def _scored_row(t_s: float, ch: str, is_anomaly: bool, fault: str = "none", score: float = 0.0):
    return {
        "timestamp": _ts(t_s),
        "channel_id": ch,
        "is_anomaly": is_anomaly,
        "predicted_fault": fault,
        "anomaly_score": score,
    }


def _label_row(t_s: float, ch: str, fault: str, severity: float = 0.8):
    return {
        "timestamp": _ts(t_s),
        "channel_id": ch,
        "fault_type": fault,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# _build_fault_windows
# ---------------------------------------------------------------------------

class TestBuildFaultWindows:
    def test_single_fault_window(self):
        labels = pd.DataFrame([
            _label_row(10, "ch_01", "overload_spike"),
            _label_row(11, "ch_01", "overload_spike"),
            _label_row(12, "ch_01", "overload_spike"),
        ])
        windows = _build_fault_windows(labels)
        assert len(windows) == 1
        assert windows[0]["fault_type"] == "overload_spike"
        assert windows[0]["channel_id"] == "ch_01"
        assert windows[0]["start"] == _ts(10)
        assert windows[0]["end"] == _ts(12)

    def test_ignores_none_labels(self):
        labels = pd.DataFrame([
            _label_row(1, "ch_01", "none"),
            _label_row(5, "ch_01", "voltage_sag"),
        ])
        windows = _build_fault_windows(labels)
        assert len(windows) == 1
        assert windows[0]["fault_type"] == "voltage_sag"

    def test_multiple_channels_and_faults(self):
        labels = pd.DataFrame([
            _label_row(10, "ch_01", "overload_spike"),
            _label_row(20, "ch_02", "thermal_drift"),
        ])
        windows = _build_fault_windows(labels)
        assert len(windows) == 2
        faults = {w["fault_type"] for w in windows}
        assert faults == {"overload_spike", "thermal_drift"}

    def test_empty_labels(self):
        labels = pd.DataFrame(columns=["timestamp", "channel_id", "fault_type", "severity"])
        windows = _build_fault_windows(labels)
        assert windows == []


# ---------------------------------------------------------------------------
# _match_fault_window
# ---------------------------------------------------------------------------

class TestMatchFaultWindow:
    def test_row_inside_window(self):
        windows = [{"channel_id": "ch_01", "fault_type": "overload_spike",
                     "start": _ts(10), "end": _ts(15)}]
        row = pd.Series({"timestamp": _ts(12), "channel_id": "ch_01"})
        assert _match_fault_window(row, windows, tolerance_s=1.0) == "overload_spike"

    def test_row_outside_window(self):
        windows = [{"channel_id": "ch_01", "fault_type": "overload_spike",
                     "start": _ts(10), "end": _ts(15)}]
        row = pd.Series({"timestamp": _ts(20), "channel_id": "ch_01"})
        assert _match_fault_window(row, windows, tolerance_s=1.0) == "none"

    def test_row_within_tolerance(self):
        windows = [{"channel_id": "ch_01", "fault_type": "overload_spike",
                     "start": _ts(10), "end": _ts(15)}]
        # 0.5s before window start, within 1s tolerance
        row = pd.Series({"timestamp": _ts(9.5), "channel_id": "ch_01"})
        assert _match_fault_window(row, windows, tolerance_s=1.0) == "overload_spike"

    def test_wrong_channel(self):
        windows = [{"channel_id": "ch_01", "fault_type": "overload_spike",
                     "start": _ts(10), "end": _ts(15)}]
        row = pd.Series({"timestamp": _ts(12), "channel_id": "ch_02"})
        assert _match_fault_window(row, windows, tolerance_s=1.0) == "none"


# ---------------------------------------------------------------------------
# evaluate — overall metrics
# ---------------------------------------------------------------------------

class TestEvaluateOverall:
    def test_perfect_detection(self):
        """All fault rows detected as anomalies, no false positives."""
        scored = pd.DataFrame([
            _scored_row(1, "ch_01", False),
            _scored_row(2, "ch_01", False),
            _scored_row(10, "ch_01", True, "overload_spike", 0.9),
            _scored_row(11, "ch_01", True, "overload_spike", 0.85),
            _scored_row(20, "ch_01", False),
        ])
        labels = pd.DataFrame([
            _label_row(10, "ch_01", "overload_spike"),
            _label_row(11, "ch_01", "overload_spike"),
        ])
        result = evaluate(scored, labels, tolerance_s=0.5)
        overall = result["overall"]
        assert overall["true_positives"] == 2
        assert overall["false_positives"] == 0
        assert overall["false_negatives"] == 0
        assert overall["true_negatives"] == 3
        assert overall["precision"] == 1.0
        assert overall["recall"] == 1.0
        assert overall["f1"] == 1.0

    def test_missed_detection(self):
        """Fault rows not detected → false negatives."""
        scored = pd.DataFrame([
            _scored_row(1, "ch_01", False),
            _scored_row(10, "ch_01", False),  # miss
            _scored_row(11, "ch_01", False),  # miss
        ])
        labels = pd.DataFrame([
            _label_row(10, "ch_01", "overload_spike"),
            _label_row(11, "ch_01", "overload_spike"),
        ])
        result = evaluate(scored, labels, tolerance_s=0.5)
        overall = result["overall"]
        assert overall["false_negatives"] == 2
        assert overall["recall"] == 0.0

    def test_false_alarm(self):
        """Anomaly flagged on nominal rows → false positive."""
        scored = pd.DataFrame([
            _scored_row(1, "ch_01", True, "noisy_sensor", 0.7),  # FP
            _scored_row(2, "ch_01", False),
        ])
        labels = pd.DataFrame(columns=["timestamp", "channel_id", "fault_type", "severity"])
        result = evaluate(scored, labels, tolerance_s=1.0)
        overall = result["overall"]
        assert overall["false_positives"] == 1
        assert overall["true_positives"] == 0
        assert overall["precision"] == 0.0

    def test_all_nominal(self):
        """No faults, no anomalies — perfect specificity."""
        scored = pd.DataFrame([
            _scored_row(1, "ch_01", False),
            _scored_row(2, "ch_01", False),
        ])
        labels = pd.DataFrame(columns=["timestamp", "channel_id", "fault_type", "severity"])
        result = evaluate(scored, labels, tolerance_s=1.0)
        overall = result["overall"]
        assert overall["true_negatives"] == 2
        assert overall["f1"] == 0.0  # no positives to measure


# ---------------------------------------------------------------------------
# evaluate — per-fault metrics
# ---------------------------------------------------------------------------

class TestEvaluatePerFault:
    def test_per_fault_recall_and_ttd(self):
        scored = pd.DataFrame([
            _scored_row(9, "ch_01", False),
            _scored_row(10, "ch_01", False),              # fault starts, not detected
            _scored_row(11, "ch_01", True, "overload_spike", 0.8),  # first detection at t=11
            _scored_row(12, "ch_01", True, "overload_spike", 0.9),
            _scored_row(20, "ch_01", False),
        ])
        labels = pd.DataFrame([
            _label_row(10, "ch_01", "overload_spike"),
            _label_row(11, "ch_01", "overload_spike"),
            _label_row(12, "ch_01", "overload_spike"),
        ])
        result = evaluate(scored, labels, tolerance_s=0.5)
        pf = result["per_fault"]
        key = "overload_spike_ch_01"
        assert key in pf
        assert pf[key]["window_rows"] == 3
        assert pf[key]["detections"] == 2
        assert pf[key]["recall"] == pytest.approx(2 / 3, abs=0.01)
        assert pf[key]["time_to_detect_s"] == pytest.approx(1.0, abs=0.1)

    def test_classification_accuracy(self):
        """Only rows with correct predicted_fault count toward classification accuracy."""
        scored = pd.DataFrame([
            _scored_row(10, "ch_01", True, "overload_spike", 0.9),
            _scored_row(11, "ch_01", True, "noisy_sensor", 0.7),   # wrong class
            _scored_row(12, "ch_01", True, "overload_spike", 0.85),
        ])
        labels = pd.DataFrame([
            _label_row(10, "ch_01", "overload_spike"),
            _label_row(11, "ch_01", "overload_spike"),
            _label_row(12, "ch_01", "overload_spike"),
        ])
        result = evaluate(scored, labels, tolerance_s=0.5)
        pf = result["per_fault"]["overload_spike_ch_01"]
        assert pf["classification_accuracy"] == pytest.approx(2 / 3, abs=0.01)

    def test_no_detection_in_window(self):
        """Fault window with zero anomaly detections → TTD is None."""
        scored = pd.DataFrame([
            _scored_row(10, "ch_01", False),
            _scored_row(11, "ch_01", False),
        ])
        labels = pd.DataFrame([
            _label_row(10, "ch_01", "voltage_sag"),
            _label_row(11, "ch_01", "voltage_sag"),
        ])
        result = evaluate(scored, labels, tolerance_s=0.5)
        pf = result["per_fault"]["voltage_sag_ch_01"]
        assert pf["time_to_detect_s"] is None
        assert pf["recall"] == 0.0

    def test_multiple_fault_types(self):
        scored = pd.DataFrame([
            _scored_row(10, "ch_01", True, "overload_spike", 0.9),
            _scored_row(30, "ch_02", True, "thermal_drift", 0.8),
        ])
        labels = pd.DataFrame([
            _label_row(10, "ch_01", "overload_spike"),
            _label_row(30, "ch_02", "thermal_drift"),
        ])
        result = evaluate(scored, labels, tolerance_s=0.5)
        assert "overload_spike_ch_01" in result["per_fault"]
        assert "thermal_drift_ch_02" in result["per_fault"]
