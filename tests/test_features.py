"""Tests for the feature engine."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from src.features.engine import FeatureEngine
from src.config.models import FeatureConfig


def _make_telemetry(n: int = 200, channel_id: str = "ch_01") -> pd.DataFrame:
    """Create a minimal telemetry DataFrame for testing."""
    t0 = datetime.now(tz=timezone.utc)
    return pd.DataFrame({
        "timestamp": [t0 + timedelta(milliseconds=i * 100) for i in range(n)],
        "channel_id": channel_id,
        "current_a": np.random.default_rng(42).normal(5.0, 0.2, n),
        "voltage_v": np.random.default_rng(42).normal(13.5, 0.05, n),
        "temperature_c": 25.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.01, n)),
        "state_on_off": True,
        "trip_flag": False,
        "overload_flag": False,
        "reset_counter": 0,
        "pwm_duty_pct": 100.0,
        "device_status": "ok",
    })


def test_compute_adds_feature_columns():
    df = _make_telemetry()
    engine = FeatureEngine(FeatureConfig(window_size=20, min_periods=5))
    result = engine.compute(df)

    expected_cols = [
        "rolling_rms_current", "rolling_mean_current", "rolling_max_current",
        "rolling_min_current", "temperature_slope", "spike_score",
        "trip_frequency", "degradation_trend",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_feature_values_reasonable():
    df = _make_telemetry(300)
    engine = FeatureEngine(FeatureConfig(window_size=20, min_periods=5))
    result = engine.compute(df)

    # RMS should be close to mean for low-noise nominal data
    valid = result["rolling_rms_current"].dropna()
    assert abs(valid.mean() - 5.0) < 1.0

    # Spike score should be low for nominal
    assert result["spike_score"].mean() < 2.0
