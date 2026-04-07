"""Tests for lifetime health tracking — histogram-based load spectra."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.config.models import EdgeConfig
from src.edge.lifetime import LifetimeHealthTracker
from src.edge.runtime import EdgeRuntime
from src.inference.pipeline import InferencePipeline
from src.schemas.telemetry import (
    CycleSummary,
    FaultType,
    HealthBand,
    LifetimeHealthState,
    LoadHistogram,
    TrendDirection,
)
from src.transport.mock_can import DataFrameTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc)


def _make_summary(
    idx: int = 0,
    stress: float = 0.05,
    anomaly_count: int = 0,
    trip_count: int = 0,
    retry_count: int = 0,
    peak_current: float = 5.0,
    peak_temp: float = 40.0,
    high_temp_dwell: float = 0.0,
    duration: float = 60.0,
    sample_count: int = 100,
) -> CycleSummary:
    """Build a CycleSummary for testing."""
    open_ts = T0 + timedelta(minutes=idx * 10)
    close_ts = open_ts + timedelta(seconds=duration)
    return CycleSummary(
        cycle_id=f"cycle_{idx:04d}",
        cycle_type="ignition",
        open_timestamp=open_ts,
        close_timestamp=close_ts,
        duration_s=duration,
        anomaly_count=anomaly_count,
        trip_count=trip_count,
        retry_count=retry_count,
        sample_count=sample_count,
        peak_current_a=peak_current,
        peak_temperature_c=peak_temp,
        high_load_dwell_s=0.0,
        high_temp_dwell_s=high_temp_dwell,
        voltage_sag_dwell_s=0.0,
        dominant_fault=FaultType.NONE,
        cycle_stress=stress,
        health_band=HealthBand.NOMINAL,
    )


# ---------------------------------------------------------------------------
# LoadHistogram tests
# ---------------------------------------------------------------------------


class TestLoadHistogram:
    def test_default_counts(self):
        h = LoadHistogram(name="test", edges=[5.0, 10.0, 15.0])
        assert h.counts == [0, 0, 0, 0]  # 3 edges → 4 bins
        assert h.total == 0

    def test_bin_index_low(self):
        h = LoadHistogram(name="test", edges=[5.0, 10.0, 15.0])
        assert h.bin_index(2.0) == 0  # below first edge

    def test_bin_index_mid(self):
        h = LoadHistogram(name="test", edges=[5.0, 10.0, 15.0])
        assert h.bin_index(7.0) == 1  # between 5 and 10

    def test_bin_index_high(self):
        h = LoadHistogram(name="test", edges=[5.0, 10.0, 15.0])
        assert h.bin_index(99.0) == 3  # above last edge

    def test_record(self):
        h = LoadHistogram(name="test", edges=[5.0, 10.0, 15.0])
        h.record(2.0)
        h.record(7.0)
        h.record(7.0)
        h.record(20.0)
        assert h.counts == [1, 2, 0, 1]
        assert h.total == 4

    def test_upper_fraction(self):
        h = LoadHistogram(name="test", edges=[5.0, 10.0, 15.0])
        # 8 in low bins, 2 in top-2 bins
        h.counts = [4, 4, 1, 1]
        assert h.upper_fraction(2) == pytest.approx(0.2)

    def test_upper_fraction_empty(self):
        h = LoadHistogram(name="test", edges=[5.0, 10.0])
        assert h.upper_fraction(2) == 0.0

    def test_serialization_round_trip(self):
        h = LoadHistogram(name="peak_current", unit="A", edges=[5.0, 10.0])
        h.record(3.0)
        h.record(12.0)
        d = h.model_dump()
        h2 = LoadHistogram.model_validate(d)
        assert h2.counts == h.counts
        assert h2.name == "peak_current"


# ---------------------------------------------------------------------------
# LifetimeHealthState schema tests
# ---------------------------------------------------------------------------


class TestLifetimeHealthStateSchema:
    def test_defaults(self):
        s = LifetimeHealthState()
        assert s.cycles_ingested == 0
        assert s.health_score == 1.0
        assert s.health_band == HealthBand.NOMINAL
        assert s.trend == TrendDirection.STABLE
        # Histograms initialized with correct bin counts
        assert len(s.peak_current_hist.counts) == 8  # 7 edges → 8 bins
        assert s.peak_current_hist.total == 0

    def test_serialization_round_trip(self):
        s = LifetimeHealthState(
            cycles_ingested=10,
            health_score=0.75,
            health_band=HealthBand.MONITOR,
            trend=TrendDirection.DEGRADING,
            last_update_timestamp=T0,
        )
        d = s.model_dump()
        assert d["health_band"] == "monitor"
        assert d["trend"] == "degrading"
        s2 = LifetimeHealthState.model_validate(d)
        assert s2.cycles_ingested == 10
        assert s2.peak_current_hist.total == 0

    def test_all_six_histograms_present(self):
        s = LifetimeHealthState()
        hist_names = [
            s.peak_current_hist.name,
            s.peak_temperature_hist.name,
            s.cycle_stress_hist.name,
            s.trips_per_cycle_hist.name,
            s.retries_per_cycle_hist.name,
            s.thermal_dwell_frac_hist.name,
        ]
        assert len(hist_names) == 6
        assert len(set(hist_names)) == 6  # all unique


class TestTrendDirection:
    def test_values(self):
        assert TrendDirection.IMPROVING.value == "improving"
        assert TrendDirection.WORSENING.value == "worsening"


# ---------------------------------------------------------------------------
# LifetimeHealthTracker unit tests
# ---------------------------------------------------------------------------


class TestLifetimeHealthTrackerBasic:
    def test_first_cycle_records_into_histograms(self):
        tracker = LifetimeHealthTracker()
        summary = _make_summary(idx=0, peak_current=8.0, peak_temp=50.0, stress=0.1)
        state = tracker.ingest(summary)

        assert state.cycles_ingested == 1
        assert state.peak_current_hist.total == 1
        assert state.peak_temperature_hist.total == 1
        assert state.cycle_stress_hist.total == 1
        assert state.last_cycle_id == "cycle_0000"

    def test_histogram_counts_accumulate(self):
        tracker = LifetimeHealthTracker()
        # Feed 10 cycles with current=5 → bin 1 (edges [2,5,8,...])
        for i in range(10):
            tracker.ingest(_make_summary(i, peak_current=6.0))
        state = tracker.state
        assert state.peak_current_hist.total == 10
        # 6.0 is between 5.0 and 8.0 → bin index 2
        assert state.peak_current_hist.counts[2] == 10

    def test_clean_cycles_stay_nominal(self):
        tracker = LifetimeHealthTracker()
        for i in range(10):
            state = tracker.ingest(_make_summary(i))
        assert state.health_band == HealthBand.NOMINAL
        assert state.health_score > 0.85

    def test_stressed_cycles_degrade_health(self):
        tracker = LifetimeHealthTracker()
        for i in range(10):
            state = tracker.ingest(
                _make_summary(
                    i,
                    stress=0.9,
                    peak_current=35.0,
                    peak_temp=170.0,
                    trip_count=25,
                    retry_count=25,
                    high_temp_dwell=55.0,
                    duration=60.0,
                )
            )
        # All values in top bins → high upper fractions → low health
        assert state.health_band in (HealthBand.DEGRADED, HealthBand.CRITICAL)
        assert state.health_score < 0.30

    def test_distribution_shift_visible(self):
        tracker = LifetimeHealthTracker()
        # 8 mild cycles
        for i in range(8):
            tracker.ingest(_make_summary(i, peak_current=4.0))
        # 2 severe cycles
        for i in range(8, 10):
            tracker.ingest(_make_summary(i, peak_current=35.0))
        state = tracker.state
        # Can query: how many cycles had peak current > 30A?
        top_bin = state.peak_current_hist.counts[-1]
        assert top_bin == 2

    def test_health_score_bounds(self):
        tracker = LifetimeHealthTracker()
        s1 = tracker.ingest(_make_summary(0))
        assert 0.0 <= s1.health_score <= 1.0

        tracker.reset()
        for i in range(20):
            s2 = tracker.ingest(
                _make_summary(
                    i,
                    stress=0.95,
                    peak_current=50.0,
                    peak_temp=180.0,
                    trip_count=30,
                    retry_count=30,
                    high_temp_dwell=58.0,
                    duration=60.0,
                )
            )
        assert 0.0 <= s2.health_score <= 1.0

    def test_reset_clears_state(self):
        tracker = LifetimeHealthTracker()
        tracker.ingest(_make_summary(0))
        tracker.reset()
        state = tracker.state
        assert state.cycles_ingested == 0
        assert state.health_score == 1.0
        assert state.peak_current_hist.total == 0


class TestLifetimeHistogramDistribution:
    def test_mixed_load_profile(self):
        """Simulate realistic fleet: mostly mild, some stressed."""
        tracker = LifetimeHealthTracker()
        # 80 mild cycles
        for i in range(80):
            tracker.ingest(_make_summary(i, peak_current=6.0, stress=0.05))
        # 15 moderate
        for i in range(80, 95):
            tracker.ingest(_make_summary(i, peak_current=14.0, stress=0.30))
        # 5 severe
        for i in range(95, 100):
            tracker.ingest(_make_summary(i, peak_current=35.0, stress=0.85))

        state = tracker.state
        assert state.cycles_ingested == 100
        # Distribution shape preserved — can query individual bins
        assert state.peak_current_hist.counts[-1] == 5  # >30A bin
        # Mostly healthy because 95% is in lower bins
        assert state.health_score > 0.50

    def test_all_in_top_bin_is_critical(self):
        tracker = LifetimeHealthTracker()
        for i in range(20):
            tracker.ingest(
                _make_summary(
                    i,
                    peak_current=50.0,
                    peak_temp=200.0,
                    stress=0.95,
                    trip_count=30,
                    retry_count=30,
                    high_temp_dwell=58.0,
                    duration=60.0,
                )
            )
        state = tracker.state
        assert state.health_band == HealthBand.CRITICAL
        assert state.health_score < 0.15


class TestLifetimeHealthTrackerTrend:
    def test_stable_trend_with_consistent_health(self):
        tracker = LifetimeHealthTracker(trend_window=5)
        for i in range(6):
            state = tracker.ingest(_make_summary(i))
        assert state.trend == TrendDirection.STABLE

    def test_degrading_trend(self):
        tracker = LifetimeHealthTracker(trend_window=5)
        # Start clean
        for i in range(3):
            tracker.ingest(_make_summary(i))
        # Then severe
        for i in range(3, 8):
            state = tracker.ingest(
                _make_summary(
                    i,
                    stress=0.9,
                    peak_current=35.0,
                    peak_temp=170.0,
                    trip_count=25,
                    retry_count=25,
                    high_temp_dwell=55.0,
                    duration=60.0,
                )
            )
        assert state.trend in (TrendDirection.DEGRADING, TrendDirection.WORSENING)

    def test_few_cycles_is_stable(self):
        tracker = LifetimeHealthTracker(trend_window=5)
        state = tracker.ingest(_make_summary(0))
        assert state.trend == TrendDirection.STABLE
        state = tracker.ingest(_make_summary(1))
        assert state.trend == TrendDirection.STABLE


class TestLifetimeHealthTrackerBandMapping:
    def test_score_to_band(self):
        assert LifetimeHealthTracker._score_to_band(0.90) == HealthBand.NOMINAL
        assert LifetimeHealthTracker._score_to_band(0.70) == HealthBand.MONITOR
        assert LifetimeHealthTracker._score_to_band(0.50) == HealthBand.DEGRADED
        assert LifetimeHealthTracker._score_to_band(0.20) == HealthBand.CRITICAL


# ---------------------------------------------------------------------------
# Integration with EdgeRuntime
# ---------------------------------------------------------------------------


class TestEdgeRuntimeLifetimeIntegration:
    def _make_telemetry(self, n: int = 200, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "timestamp": [T0 + timedelta(milliseconds=i * 100) for i in range(n)],
                "channel_id": "ch_01",
                "current_a": rng.normal(5.0, 0.1, n),
                "voltage_v": 13.5,
                "temperature_c": 40.0 + rng.normal(0, 0.5, n),
                "state_on_off": True,
                "trip_flag": False,
                "overload_flag": False,
                "reset_counter": 0,
                "pwm_duty_pct": 100.0,
            }
        )

    def test_lifetime_disabled_by_default(self):
        df = self._make_telemetry(100)
        transport = DataFrameTransport(df)
        pipeline = InferencePipeline()
        cfg = EdgeConfig(batch_size=25)
        runtime = EdgeRuntime(transport, pipeline, cfg)
        assert runtime._lifetime_tracker is None

    def test_lifetime_requires_cycle_tracking(self):
        cfg = EdgeConfig(
            lifetime_tracking_enabled=True,
            cycle_tracking_enabled=False,
        )
        runtime = EdgeRuntime(
            DataFrameTransport(self._make_telemetry(100)),
            InferencePipeline(),
            cfg,
        )
        assert runtime._lifetime_tracker is None

    def test_lifetime_enabled_produces_state(self):
        df = self._make_telemetry(200)
        transport = DataFrameTransport(df)
        pipeline = InferencePipeline()
        cfg = EdgeConfig(
            batch_size=50,
            cycle_tracking_enabled=True,
            lifetime_tracking_enabled=True,
            cycle_boundary_column=None,
        )
        runtime = EdgeRuntime(transport, pipeline, cfg)
        runtime.run()
        assert len(runtime.cycle_summaries) >= 1
        state = runtime._lifetime_tracker.state
        assert state.cycles_ingested >= 1
        assert state.peak_current_hist.total >= 1

    def test_lifetime_with_boundary_cycles(self):
        rng = np.random.default_rng(77)
        rows = []
        for i in range(300):
            on = not (80 <= i < 100) and not (200 <= i < 220)
            rows.append(
                {
                    "timestamp": T0 + timedelta(milliseconds=i * 100),
                    "channel_id": "ch_01",
                    "current_a": rng.normal(5.0, 0.1),
                    "voltage_v": 13.5,
                    "temperature_c": 40.0,
                    "state_on_off": on,
                    "trip_flag": False,
                    "overload_flag": False,
                    "reset_counter": 0,
                    "pwm_duty_pct": 100.0,
                }
            )
        df = pd.DataFrame(rows)
        transport = DataFrameTransport(df)
        pipeline = InferencePipeline()
        cfg = EdgeConfig(
            batch_size=50,
            cycle_tracking_enabled=True,
            lifetime_tracking_enabled=True,
            cycle_boundary_column="state_on_off",
        )
        runtime = EdgeRuntime(transport, pipeline, cfg)
        runtime.run()

        assert len(runtime.cycle_summaries) >= 2
        state = runtime._lifetime_tracker.state
        assert state.cycles_ingested >= 2
        # Histograms populated
        assert state.peak_current_hist.total >= 2
        assert state.cycle_stress_hist.total >= 2
