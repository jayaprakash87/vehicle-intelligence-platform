"""Tests for the ingestion normalizer."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from src.ingestion.normalizer import Normalizer


def _make_raw(n: int = 100, channel_id: str = "ch_01") -> pd.DataFrame:
    t0 = datetime.now(tz=timezone.utc)
    return pd.DataFrame({
        "timestamp": [t0 + timedelta(milliseconds=i * 100) for i in range(n)],
        "channel_id": channel_id,
        "current_a": np.random.default_rng(42).normal(5.0, 0.2, n),
        "voltage_v": np.random.default_rng(42).normal(13.5, 0.05, n),
        "temperature_c": 25.0 + np.random.default_rng(42).normal(0, 0.5, n),
        "state_on_off": True,
        "trip_flag": False,
        "overload_flag": False,
        "reset_counter": 0,
        "pwm_duty_pct": 100.0,
        "device_status": "ok",
    })


def test_normalize_sorts_by_timestamp():
    df = _make_raw()
    # Reverse order
    df = df.iloc[::-1].reset_index(drop=True)
    norm = Normalizer()
    result = norm.normalize(df)
    assert result["timestamp"].is_monotonic_increasing


def test_ffill_limited_to_5():
    """NaN gaps > 5 rows should NOT be fully filled."""
    df = _make_raw(50)
    # Insert a 10-row NaN gap
    df.loc[20:29, "current_a"] = np.nan
    norm = Normalizer()
    result = norm.normalize(df)
    # First 5 NaN rows get filled, remaining 5 stay NaN (stale)
    stale = result["_stale_current_a"]
    assert stale.sum() > 0, "Long NaN gap should leave stale rows"


def test_missing_rate_computed_before_ffill():
    """missing_rate should reflect original NaN pattern, not post-ffill state."""
    df = _make_raw(100)
    # Set 30% of rows to NaN
    rng = np.random.default_rng(0)
    mask = rng.random(100) < 0.3
    df.loc[mask, "current_a"] = np.nan
    norm = Normalizer()
    result = norm.normalize(df)
    # missing_rate should be > 0 somewhere
    assert result["missing_rate"].max() > 0.1


def test_clipping_bounds():
    df = _make_raw(10)
    df.loc[0, "current_a"] = 500.0
    df.loc[1, "current_a"] = -10.0
    df.loc[2, "voltage_v"] = 100.0
    df.loc[3, "temperature_c"] = 200.0
    df.loc[4, "pwm_duty_pct"] = 150.0
    norm = Normalizer()
    result = norm.normalize(df)
    assert result["current_a"].max() <= 200
    assert result["current_a"].min() >= -1
    assert result["voltage_v"].max() <= 60
    assert result["temperature_c"].max() <= 150
    assert result["pwm_duty_pct"].max() <= 100


def test_boolean_coercion():
    df = _make_raw(10)
    df["trip_flag"] = 0  # int, not bool
    df["overload_flag"] = 1
    norm = Normalizer()
    result = norm.normalize(df)
    assert result["trip_flag"].dtype == bool
    assert result["overload_flag"].dtype == bool


def test_validate_record_valid():
    norm = Normalizer()
    row = {
        "timestamp": datetime.now(tz=timezone.utc),
        "channel_id": "ch_01",
        "current_a": 5.0,
        "voltage_v": 13.5,
        "temperature_c": 25.0,
        "state_on_off": True,
    }
    result = norm.validate_record(row)
    assert result is not None
    assert result.channel_id == "ch_01"


def test_validate_record_invalid():
    norm = Normalizer()
    result = norm.validate_record({"bad": "data"})
    assert result is None


def test_normalize_unsorted_multi_channel():
    """Normalizer should handle interleaved multi-channel data."""
    df1 = _make_raw(50, "ch_01")
    df2 = _make_raw(50, "ch_02")
    df = pd.concat([df2, df1], ignore_index=True)  # ch_02 first
    norm = Normalizer()
    result = norm.normalize(df)
    assert len(result) == 100
    assert result["timestamp"].is_monotonic_increasing
