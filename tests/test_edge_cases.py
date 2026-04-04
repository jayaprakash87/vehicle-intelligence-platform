"""Edge-case and robustness tests — empty data, single row, all-NaN, wrong columns."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.features.engine import FeatureEngine
from src.ingestion.normalizer import Normalizer
from src.models.anomaly import AnomalyDetector, ANOMALY_FEATURES
from src.config.models import FeatureConfig, ModelConfig


def test_normalizer_empty_df():
    norm = Normalizer()
    df = pd.DataFrame(columns=[
        "timestamp", "channel_id", "current_a", "voltage_v",
        "temperature_c", "state_on_off", "trip_flag", "overload_flag",
        "reset_counter", "pwm_duty_pct",
    ])
    result = norm.normalize(df)
    assert len(result) == 0


def test_normalizer_single_row():
    norm = Normalizer()
    df = pd.DataFrame([{
        "timestamp": datetime.now(tz=timezone.utc),
        "channel_id": "ch_01",
        "current_a": 5.0,
        "voltage_v": 13.5,
        "temperature_c": 40.0,
        "state_on_off": True,
        "trip_flag": False,
        "overload_flag": False,
        "reset_counter": 0,
        "pwm_duty_pct": 100.0,
    }])
    result = norm.normalize(df)
    assert len(result) == 1


def test_feature_engine_single_row():
    """Feature engine should not crash on a single row — NaNs expected."""
    engine = FeatureEngine(FeatureConfig(window_size=5, min_periods=1))
    df = pd.DataFrame([{
        "timestamp": datetime.now(tz=timezone.utc),
        "channel_id": "ch_01",
        "current_a": 5.0,
        "voltage_v": 13.5,
        "temperature_c": 40.0,
        "state_on_off": True,
        "trip_flag": False,
        "overload_flag": False,
        "reset_counter": 0,
        "pwm_duty_pct": 100.0,
    }])
    result = engine.compute(df)
    assert "rolling_rms_current" in result.columns


def test_anomaly_prepare_missing_cols():
    """_prepare should raise ValueError when feature columns are missing."""
    detector = AnomalyDetector()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="Missing feature columns"):
        detector._prepare(df)


def test_anomaly_prepare_all_nan():
    """_prepare should raise ValueError when all feature rows are NaN."""
    detector = AnomalyDetector()
    df = pd.DataFrame({col: [np.nan] * 5 for col in ANOMALY_FEATURES})
    with pytest.raises(ValueError, match="No valid rows"):
        detector._prepare(df)


def test_anomaly_score_untrained_raises():
    detector = AnomalyDetector()
    df = pd.DataFrame({col: [1.0, 2.0] for col in ANOMALY_FEATURES})
    with pytest.raises(RuntimeError, match="not trained"):
        detector.score(df)


def test_feature_engine_all_nan_current():
    """Feature engine should handle all-NaN current without crashing."""
    engine = FeatureEngine(FeatureConfig(window_size=5, min_periods=1))
    n = 10
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="100ms"),
        "channel_id": "ch_01",
        "current_a": np.nan,
        "voltage_v": 13.5,
        "temperature_c": 40.0,
        "state_on_off": True,
        "trip_flag": False,
        "overload_flag": False,
        "reset_counter": 0,
        "pwm_duty_pct": 100.0,
    })
    result = engine.compute(df)
    assert len(result) == n


def test_feature_engine_large_values():
    """Extreme values should not cause overflow or crash."""
    engine = FeatureEngine(FeatureConfig(window_size=5, min_periods=1))
    n = 20
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="100ms"),
        "channel_id": "ch_01",
        "current_a": 1e6,
        "voltage_v": 1e6,
        "temperature_c": 1e6,
        "state_on_off": True,
        "trip_flag": False,
        "overload_flag": False,
        "reset_counter": 0,
        "pwm_duty_pct": 100.0,
    })
    result = engine.compute(df)
    assert len(result) == n
    assert np.isfinite(result["rolling_rms_current"]).all()


def test_normalizer_non_boolean_trip_flag():
    """Non-boolean types for trip_flag should be coerced."""
    norm = Normalizer()
    n = 5
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="100ms"),
        "channel_id": "ch_01",
        "current_a": [5.0] * n,
        "voltage_v": [13.5] * n,
        "temperature_c": [40.0] * n,
        "state_on_off": [1, 0, 1, "true", "false"],
        "trip_flag": [0, 1, 0, 1, 0],
        "overload_flag": [0, 0, 0, 0, 0],
        "reset_counter": [0] * n,
        "pwm_duty_pct": [100.0] * n,
    })
    result = norm.normalize(df)
    assert result["trip_flag"].dtype == bool
    assert result["state_on_off"].dtype == bool
