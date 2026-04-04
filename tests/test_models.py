"""Tests for anomaly detection and fault classification."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.config.models import FeatureConfig, ModelConfig
from src.features.engine import FeatureEngine
from src.models.anomaly import AnomalyDetector, RulesFaultClassifier
from src.schemas.telemetry import FaultType


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
