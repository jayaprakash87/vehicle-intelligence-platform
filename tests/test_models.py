"""Tests for anomaly detection and fault classification."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.config.models import FeatureConfig, ModelConfig
from src.features.engine import FeatureEngine
from src.models.anomaly import AnomalyDetector, RulesFaultClassifier
from src.schemas.telemetry import FaultType, ProtectionEvent


def _make_featured_df(n: int = 500) -> pd.DataFrame:
    """Generate a feature-enriched DataFrame for model testing."""
    rng = np.random.default_rng(42)
    t0 = datetime.now(tz=timezone.utc)
    df = pd.DataFrame({
        "timestamp": [t0 + timedelta(milliseconds=i * 100) for i in range(n)],
        "channel_id": "ch_01",
        "current_a": rng.normal(5.0, 0.2, n),
        "voltage_v": rng.normal(13.5, 0.05, n),
        "temperature_c": 25.0 + np.cumsum(rng.normal(0, 0.01, n)),
        "state_on_off": True,
        "trip_flag": False,
        "overload_flag": False,
        "reset_counter": 0,
        "pwm_duty_pct": 100.0,
        "device_status": "ok",
    })
    engine = FeatureEngine(FeatureConfig(window_size=20, min_periods=5))
    return engine.compute(df)


def test_anomaly_detector_train_predict():
    df = _make_featured_df()
    detector = AnomalyDetector(ModelConfig(anomaly_contamination=0.05))
    detector.train(df)

    scores = detector.score(df)
    assert len(scores) == len(df)
    assert scores.min() >= 0
    assert scores.max() <= 1.0

    preds = detector.predict(df)
    assert preds.dtype == bool


def test_anomaly_detector_save_load(tmp_path):
    df = _make_featured_df()
    detector = AnomalyDetector(ModelConfig())
    detector.train(df)
    detector.save(tmp_path)

    loaded = AnomalyDetector(ModelConfig())
    loaded.load(tmp_path)
    s1 = detector.score(df)
    s2 = loaded.score(df)
    pd.testing.assert_series_equal(s1, s2)


def test_rules_classifier_overload():
    clf = RulesFaultClassifier()
    row = {
        "spike_score": 5.0,
        "trip_flag": True,
        "overload_flag": True,
        "trip_frequency": 0,
        "temperature_slope": 0,
        "degradation_trend": 0,
        "rolling_rms_current": 15.0,
        "current_a": 18.0,
        "voltage_v": 13.5,
    }
    fault, conf, causes = clf.classify(row)
    assert fault == FaultType.OVERLOAD_SPIKE
    assert conf > 0


def test_rules_classifier_nominal():
    clf = RulesFaultClassifier()
    row = {
        "spike_score": 0.1,
        "trip_flag": False,
        "overload_flag": False,
        "trip_frequency": 0,
        "temperature_slope": 0.01,
        "degradation_trend": 0.0,
        "rolling_rms_current": 5.0,
        "current_a": 5.0,
        "voltage_v": 13.5,
    }
    fault, conf, causes = clf.classify(row)
    assert fault == FaultType.NONE


# ---------------------------------------------------------------------------
# Protection-event-aware classification
# ---------------------------------------------------------------------------

def test_classifier_latch_off_from_scp():
    """Latch-off with preceding SCP trips should reference short-circuit."""
    clf = RulesFaultClassifier()
    row = {
        "spike_score": 6.0,
        "trip_flag": True,
        "overload_flag": True,
        "trip_frequency": 3,
        "temperature_slope": 0.1,
        "degradation_trend": 0.0,
        "rolling_rms_current": 15.0,
        "current_a": 0.001,
        "voltage_v": 13.5,
        "protection_event": ProtectionEvent.LATCH_OFF.value,
        "scp_count": 3,
        "i2t_count": 0,
        "latch_off_count": 1,
        "thermal_shutdown_count": 0,
    }
    fault, conf, causes = clf.classify(row)
    assert fault == FaultType.OVERLOAD_SPIKE
    assert conf >= 0.8
    assert any("latched off" in c.lower() for c in causes)
    assert any("scp" in c.lower() or "short-circuit" in c.lower() for c in causes)


def test_classifier_latch_off_from_i2t():
    """Latch-off with preceding I2t trips should reference sustained overload."""
    clf = RulesFaultClassifier()
    row = {
        "spike_score": 5.0,
        "trip_flag": True,
        "overload_flag": True,
        "trip_frequency": 3,
        "temperature_slope": 0.1,
        "degradation_trend": 0.0,
        "rolling_rms_current": 12.0,
        "current_a": 0.001,
        "voltage_v": 13.5,
        "protection_event": ProtectionEvent.LATCH_OFF.value,
        "scp_count": 0,
        "i2t_count": 3,
        "latch_off_count": 1,
        "thermal_shutdown_count": 0,
    }
    fault, conf, causes = clf.classify(row)
    assert fault == FaultType.OVERLOAD_SPIKE
    assert any("overload" in c.lower() or "i²t" in c.lower() for c in causes)


def test_classifier_thermal_shutdown():
    """Thermal shutdown event should classify as THERMAL_DRIFT."""
    clf = RulesFaultClassifier()
    row = {
        "spike_score": 1.0,
        "trip_flag": True,
        "overload_flag": False,
        "trip_frequency": 0,
        "temperature_slope": 0.5,
        "degradation_trend": 0.0,
        "rolling_rms_current": 10.0,
        "current_a": 0.001,
        "voltage_v": 13.5,
        "protection_event": ProtectionEvent.THERMAL_SHUTDOWN.value,
        "scp_count": 0,
        "i2t_count": 0,
        "latch_off_count": 0,
        "thermal_shutdown_count": 1,
    }
    fault, conf, causes = clf.classify(row)
    assert fault == FaultType.THERMAL_DRIFT
    assert conf >= 0.7
    assert any("thermal" in c.lower() for c in causes)


def test_classifier_scp_overload():
    """SCP event with spike should give SCP-specific cause text."""
    clf = RulesFaultClassifier()
    row = {
        "spike_score": 5.0,
        "trip_flag": True,
        "overload_flag": True,
        "trip_frequency": 0,
        "temperature_slope": 0.0,
        "degradation_trend": 0.0,
        "rolling_rms_current": 15.0,
        "current_a": 18.0,
        "voltage_v": 13.5,
        "protection_event": ProtectionEvent.SCP.value,
        "scp_count": 1,
        "i2t_count": 0,
        "latch_off_count": 0,
        "thermal_shutdown_count": 0,
    }
    fault, conf, causes = clf.classify(row)
    assert fault == FaultType.OVERLOAD_SPIKE
    assert any("short-circuit" in c.lower() or "wiring" in c.lower() for c in causes)


def test_classifier_i2t_overload():
    """I2T event with spike should give I2T-specific cause text."""
    clf = RulesFaultClassifier()
    row = {
        "spike_score": 5.0,
        "trip_flag": True,
        "overload_flag": True,
        "trip_frequency": 0,
        "temperature_slope": 0.0,
        "degradation_trend": 0.0,
        "rolling_rms_current": 15.0,
        "current_a": 18.0,
        "voltage_v": 13.5,
        "protection_event": ProtectionEvent.I2T.value,
        "scp_count": 0,
        "i2t_count": 1,
        "latch_off_count": 0,
        "thermal_shutdown_count": 0,
    }
    fault, conf, causes = clf.classify(row)
    assert fault == FaultType.OVERLOAD_SPIKE
    assert any("i²t" in c.lower() or "energy" in c.lower() for c in causes)


def test_classifier_intermittent_mixed_trips():
    """Intermittent overload with mixed SCP+I2T should note both mechanisms."""
    clf = RulesFaultClassifier()
    row = {
        "spike_score": 2.0,
        "trip_flag": False,
        "overload_flag": True,
        "trip_frequency": 4,
        "temperature_slope": 0.0,
        "degradation_trend": 0.0,
        "rolling_rms_current": 10.0,
        "current_a": 12.0,
        "voltage_v": 13.5,
        "protection_event": ProtectionEvent.NONE.value,
        "scp_count": 2,
        "i2t_count": 1,
        "latch_off_count": 0,
        "thermal_shutdown_count": 0,
    }
    fault, conf, causes = clf.classify(row)
    assert fault == FaultType.INTERMITTENT_OVERLOAD
    assert any("mixed" in c.lower() or "scp" in c.lower() for c in causes)
