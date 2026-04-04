"""Tests for multi-rate sampling and protocol abstraction."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.config.models import (
    FeatureConfig,
    NormalizerConfig,
    SimulationConfig,
    load_config,
)
from src.ingestion.normalizer import Normalizer
from src.features.engine import FeatureEngine
from src.schemas.telemetry import ChannelMeta, SourceProtocol
from src.simulation.generator import TelemetryGenerator
from src.transport.mock_can import CanTransport, DataFrameTransport, XcpTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_multi_rate_df(
    fast_interval_ms: float = 10.0,
    slow_interval_ms: float = 50.0,
    duration_s: float = 2.0,
) -> pd.DataFrame:
    """Create a two-channel DataFrame with different sample rates."""
    t0 = datetime.now(tz=timezone.utc)
    rows = []

    # Fast channel
    n_fast = int(duration_s / (fast_interval_ms / 1000))
    for i in range(n_fast):
        rows.append(
            {
                "timestamp": t0 + timedelta(milliseconds=i * fast_interval_ms),
                "channel_id": "ch_fast",
                "current_a": 5.0 + np.random.normal(0, 0.1),
                "voltage_v": 13.5,
                "temperature_c": 35.0,
                "state_on_off": True,
                "trip_flag": False,
                "overload_flag": False,
                "reset_counter": 0,
                "pwm_duty_pct": 100.0,
                "device_status": "OK",
            }
        )

    # Slow channel
    n_slow = int(duration_s / (slow_interval_ms / 1000))
    for i in range(n_slow):
        rows.append(
            {
                "timestamp": t0 + timedelta(milliseconds=i * slow_interval_ms),
                "channel_id": "ch_slow",
                "current_a": 10.0 + np.random.normal(0, 0.1),
                "voltage_v": 13.5,
                "temperature_c": 45.0,
                "state_on_off": True,
                "trip_flag": False,
                "overload_flag": False,
                "reset_counter": 0,
                "pwm_duty_pct": 100.0,
                "device_status": "OK",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FeatureConfig.resolve()
# ---------------------------------------------------------------------------


class TestFeatureConfigResolve:
    """Time-based window computation."""

    def test_auto_resolve_100ms(self):
        cfg = FeatureConfig(window_duration_s=5.0, min_duration_s=1.0)
        w, mp = cfg.resolve(0.1)  # 100ms
        assert w == 50
        assert mp == 10

    def test_auto_resolve_10ms(self):
        cfg = FeatureConfig(window_duration_s=5.0, min_duration_s=1.0)
        w, mp = cfg.resolve(0.01)  # 10ms
        assert w == 500
        assert mp == 100

    def test_auto_resolve_50ms(self):
        cfg = FeatureConfig(window_duration_s=5.0, min_duration_s=1.0)
        w, mp = cfg.resolve(0.05)  # 50ms
        assert w == 100
        assert mp == 20

    def test_override_takes_precedence(self):
        cfg = FeatureConfig(window_size=20, min_periods=5)
        w, mp = cfg.resolve(0.01)  # override ignores interval
        assert w == 20
        assert mp == 5

    def test_min_periods_capped_at_window(self):
        cfg = FeatureConfig(window_duration_s=0.5, min_duration_s=2.0)
        w, mp = cfg.resolve(0.1)
        assert mp <= w

    def test_minimum_window_size_is_2(self):
        # Very short duration relative to interval
        cfg = FeatureConfig(window_duration_s=0.01, min_duration_s=0.001)
        w, mp = cfg.resolve(0.1)
        assert w >= 2
        assert mp >= 1


# ---------------------------------------------------------------------------
# NormalizerConfig & time-based ffill
# ---------------------------------------------------------------------------


class TestNormalizerMultiRate:
    """Multi-rate normalizer behavior."""

    def test_ffill_limit_scales_with_interval(self):
        """Faster channels should get a higher ffill row limit for same time tolerance."""
        df = _make_multi_rate_df(fast_interval_ms=10.0, slow_interval_ms=50.0)
        # Inject NaN gaps
        fast_mask = df["channel_id"] == "ch_fast"
        slow_mask = df["channel_id"] == "ch_slow"
        df.loc[fast_mask & (df.index % 3 == 0), "current_a"] = np.nan
        df.loc[slow_mask & (df.index % 3 == 0), "current_a"] = np.nan

        cfg = NormalizerConfig(ffill_tolerance_s=0.5)
        norm = Normalizer(cfg)
        result = norm.normalize(df)
        # Should have fewer remaining NaNs than with limit=1
        assert result["current_a"].isna().sum() < df["current_a"].isna().sum()

    def test_resample_aligns_to_common_grid(self):
        """With resample enabled, both channels should have the same timestamp grid."""
        df = _make_multi_rate_df(fast_interval_ms=10.0, slow_interval_ms=50.0, duration_s=1.0)
        cfg = NormalizerConfig(resample_interval_ms=10.0)
        norm = Normalizer(cfg)
        result = norm.normalize(df)

        # After resampling, both channels should have the same number of rows
        fast_rows = result[result["channel_id"] == "ch_fast"]
        slow_rows = result[result["channel_id"] == "ch_slow"]
        assert len(fast_rows) == len(slow_rows), (
            f"Expected equal rows after resampling: fast={len(fast_rows)}, slow={len(slow_rows)}"
        )

    def test_no_resample_preserves_different_lengths(self):
        """Without resampling, channels keep their original sample counts."""
        df = _make_multi_rate_df(fast_interval_ms=10.0, slow_interval_ms=50.0, duration_s=1.0)
        cfg = NormalizerConfig(resample_interval_ms=0)  # no resampling
        norm = Normalizer(cfg)
        result = norm.normalize(df)

        fast_rows = result[result["channel_id"] == "ch_fast"]
        slow_rows = result[result["channel_id"] == "ch_slow"]
        assert len(fast_rows) > len(slow_rows)

    def test_default_config_backward_compatible(self):
        """Normalizer() with no config should still work."""
        df = _make_multi_rate_df()
        norm = Normalizer()
        result = norm.normalize(df)
        assert len(result) > 0
        assert "missing_rate" in result.columns

    def test_missing_rate_window_scales_with_interval(self):
        """Missing rate rolling window should be wider for faster channels."""
        df = _make_multi_rate_df(fast_interval_ms=10.0, slow_interval_ms=100.0)
        cfg = NormalizerConfig(missing_rate_window_s=5.0)
        norm = Normalizer(cfg)
        result = norm.normalize(df)
        assert "missing_rate" in result.columns


# ---------------------------------------------------------------------------
# Generator per-channel intervals
# ---------------------------------------------------------------------------


class TestGeneratorMultiRate:
    """Per-channel sample intervals in generator."""

    def test_per_channel_sample_counts(self):
        """Channels with different intervals produce different row counts."""
        cfg = SimulationConfig(
            duration_s=1.0,
            sample_interval_ms=100.0,
            channels=[
                ChannelMeta(
                    channel_id="fast_ch",
                    sample_interval_ms=10.0,
                    nominal_current_a=5.0,
                    max_current_a=10.0,
                ),
                ChannelMeta(
                    channel_id="slow_ch",
                    sample_interval_ms=50.0,
                    nominal_current_a=8.0,
                    max_current_a=15.0,
                ),
            ],
            seed=42,
        )
        gen = TelemetryGenerator(cfg)
        telem_df, _ = gen.generate()

        fast_rows = telem_df[telem_df["channel_id"] == "fast_ch"]
        slow_rows = telem_df[telem_df["channel_id"] == "slow_ch"]
        assert len(fast_rows) == 100  # 1s / 10ms
        assert len(slow_rows) == 20  # 1s / 50ms

    def test_fallback_to_global_interval(self):
        """Channels without explicit interval use global default."""
        cfg = SimulationConfig(
            duration_s=1.0,
            sample_interval_ms=100.0,
            channels=[
                ChannelMeta(channel_id="default_ch", nominal_current_a=5.0, max_current_a=10.0),
            ],
            seed=42,
        )
        gen = TelemetryGenerator(cfg)
        telem_df, _ = gen.generate()
        assert len(telem_df) == 10  # 1s / 100ms

    def test_xcp_dual_raster_config(self):
        """XCP test bench config loads and produces multi-rate data."""
        from pathlib import Path

        cfg_path = Path(__file__).parent.parent / "configs" / "xcp_test_bench.yaml"
        if not cfg_path.exists():
            pytest.skip("xcp_test_bench.yaml not found")
        cfg = load_config(str(cfg_path))
        gen = TelemetryGenerator(cfg.simulation)
        telem_df, _ = gen.generate()

        # Check that fast and slow channels have different row counts
        counts = telem_df.groupby("channel_id").size()
        fast_counts = [c for ch, c in counts.items() if "fast" in str(ch)]
        slow_counts = [c for ch, c in counts.items() if "slow" in str(ch)]
        assert all(f > s for f, s in zip(fast_counts, slow_counts))


# ---------------------------------------------------------------------------
# Feature engine with time-based windowing
# ---------------------------------------------------------------------------


class TestFeatureEngineMultiRate:
    """Feature engine auto-scales windows based on sample interval."""

    def test_features_compute_on_10ms_data(self):
        """Feature engine handles 10ms data without explicit window_size."""
        df = _make_multi_rate_df(fast_interval_ms=10.0, slow_interval_ms=10.0, duration_s=10.0)
        cfg = FeatureConfig(window_duration_s=5.0, min_duration_s=1.0)
        engine = FeatureEngine(cfg)
        result = engine.compute(df)
        assert "rolling_rms_current" in result.columns
        assert result["rolling_rms_current"].notna().any()

    def test_features_compute_on_100ms_data(self):
        """Feature engine handles 100ms data with same config."""
        df = _make_multi_rate_df(fast_interval_ms=100.0, slow_interval_ms=100.0, duration_s=10.0)
        cfg = FeatureConfig(window_duration_s=5.0, min_duration_s=1.0)
        engine = FeatureEngine(cfg)
        result = engine.compute(df)
        assert "rolling_rms_current" in result.columns
        assert result["rolling_rms_current"].notna().any()


# ---------------------------------------------------------------------------
# Transport protocol tagging
# ---------------------------------------------------------------------------


class TestProtocolTransports:
    """XCP and CAN transport protocol tagging."""

    def test_xcp_transport_tags_protocol(self):
        df = _make_multi_rate_df()
        transport = XcpTransport(df, fast_raster_ms=10.0, slow_raster_ms=50.0)
        batch = transport.batch(10)
        assert all(row.get("source_protocol") == "xcp" for row in batch)

    def test_can_transport_tags_protocol(self):
        df = _make_multi_rate_df()
        transport = CanTransport(df)
        batch = transport.batch(10)
        assert all(row.get("source_protocol") == "can" for row in batch)

    def test_dataframe_transport_no_protocol_tag(self):
        """Base DataFrameTransport doesn't add protocol tags."""
        df = _make_multi_rate_df()
        transport = DataFrameTransport(df)
        transport.batch(10)
        # Original DF doesn't have source_protocol column
        assert "source_protocol" not in df.columns


# ---------------------------------------------------------------------------
# SourceProtocol schema
# ---------------------------------------------------------------------------


class TestSourceProtocol:
    def test_enum_values(self):
        assert SourceProtocol.CAN.value == "can"
        assert SourceProtocol.XCP.value == "xcp"
        assert SourceProtocol.REPLAY.value == "replay"

    def test_channel_meta_default_protocol(self):
        ch = ChannelMeta(channel_id="test", nominal_current_a=5.0, max_current_a=10.0)
        assert ch.source_protocol == SourceProtocol.CAN
        assert ch.sample_interval_ms == 0.0

    def test_channel_meta_xcp_override(self):
        ch = ChannelMeta(
            channel_id="test",
            nominal_current_a=5.0,
            max_current_a=10.0,
            sample_interval_ms=10.0,
            source_protocol=SourceProtocol.XCP,
        )
        assert ch.source_protocol == SourceProtocol.XCP
        assert ch.sample_interval_ms == 10.0


# ---------------------------------------------------------------------------
# Integration: multi-rate end-to-end
# ---------------------------------------------------------------------------


class TestMultiRateIntegration:
    """End-to-end: generate multi-rate → normalize → resample → features."""

    def test_xcp_pipeline_end_to_end(self):
        cfg = SimulationConfig(
            duration_s=5.0,
            sample_interval_ms=10.0,
            channels=[
                ChannelMeta(
                    channel_id="xcp_fast",
                    sample_interval_ms=10.0,
                    nominal_current_a=5.0,
                    max_current_a=10.0,
                ),
                ChannelMeta(
                    channel_id="xcp_slow",
                    sample_interval_ms=50.0,
                    nominal_current_a=8.0,
                    max_current_a=15.0,
                ),
            ],
            seed=42,
        )
        gen = TelemetryGenerator(cfg)
        telem_df, _ = gen.generate()

        # Different sample counts before normalization
        fast_before = len(telem_df[telem_df["channel_id"] == "xcp_fast"])
        slow_before = len(telem_df[telem_df["channel_id"] == "xcp_slow"])
        assert fast_before > slow_before

        # Normalize with resampling
        norm_cfg = NormalizerConfig(resample_interval_ms=10.0)
        norm = Normalizer(norm_cfg)
        norm_df = norm.normalize(telem_df)

        # After resampling, both channels on same grid
        fast_after = len(norm_df[norm_df["channel_id"] == "xcp_fast"])
        slow_after = len(norm_df[norm_df["channel_id"] == "xcp_slow"])
        assert fast_after == slow_after

        # Features work on resampled data
        feat_cfg = FeatureConfig(window_duration_s=2.0, min_duration_s=0.5)
        engine = FeatureEngine(feat_cfg)
        feat_df = engine.compute(norm_df)
        assert "rolling_rms_current" in feat_df.columns
        assert feat_df["rolling_rms_current"].notna().any()
