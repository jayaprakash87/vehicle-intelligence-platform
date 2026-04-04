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


# ---------------------------------------------------------------------------
# Protection event features
# ---------------------------------------------------------------------------

def test_protection_event_features_present():
    """When protection_event column exists, event features should be added."""
    from src.schemas.telemetry import ProtectionEvent
    df = _make_telemetry(200)
    np.random.default_rng(99)
    # Inject some SCP and I2T events
    events = [ProtectionEvent.NONE.value] * 200
    for i in range(50, 60):
        events[i] = ProtectionEvent.SCP.value
    for i in range(80, 90):
        events[i] = ProtectionEvent.I2T.value
    df["protection_event"] = events
    engine = FeatureEngine(FeatureConfig(window_size=20, min_periods=5))
    result = engine.compute(df)
    for col in ("protection_event_rate", "scp_count", "i2t_count",
                "latch_off_count", "thermal_shutdown_count"):
        assert col in result.columns, f"Missing column: {col}"


def test_protection_event_counts_nonzero():
    """Rolling counts should reflect injected protection events."""
    from src.schemas.telemetry import ProtectionEvent
    df = _make_telemetry(200)
    events = [ProtectionEvent.NONE.value] * 200
    for i in range(50, 60):
        events[i] = ProtectionEvent.SCP.value
    df["protection_event"] = events
    engine = FeatureEngine(FeatureConfig(window_size=20, min_periods=5))
    result = engine.compute(df)
    # After the SCP burst, scp_count should be > 0 somewhere
    assert result["scp_count"].max() > 0
    # i2t_count should remain 0
    assert result["i2t_count"].max() == 0


def test_protection_event_rate_range():
    """protection_event_rate should be between 0 and 1."""
    from src.schemas.telemetry import ProtectionEvent
    df = _make_telemetry(200)
    events = [ProtectionEvent.NONE.value] * 200
    for i in range(100, 120):
        events[i] = ProtectionEvent.THERMAL_SHUTDOWN.value
    df["protection_event"] = events
    engine = FeatureEngine(FeatureConfig(window_size=20, min_periods=5))
    result = engine.compute(df)
    assert result["protection_event_rate"].min() >= 0.0
    assert result["protection_event_rate"].max() <= 1.0


def test_no_protection_event_column_still_works():
    """Feature engine should work without protection_event column (backward compat)."""
    df = _make_telemetry(200)
    assert "protection_event" not in df.columns
    engine = FeatureEngine(FeatureConfig(window_size=20, min_periods=5))
    result = engine.compute(df)
    assert "rolling_rms_current" in result.columns
    # Protection-event-derived features should be absent
    assert "scp_count" not in result.columns
