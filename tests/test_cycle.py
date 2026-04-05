"""Tests for cycle tracking — CycleAccumulator + CycleSummary schema."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.config.models import EdgeConfig
from src.edge.cycle import CycleAccumulator
from src.inference.pipeline import InferencePipeline
from src.schemas.telemetry import CycleSummary, FaultType, HealthBand
from src.transport.mock_can import DataFrameTransport
from src.edge.runtime import EdgeRuntime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc)


def _scored_df(
    n: int = 100,
    anomaly_frac: float = 0.0,
    trip_frac: float = 0.0,
    state_on: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a scored DataFrame mimicking InferencePipeline output."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "timestamp": [T0 + timedelta(milliseconds=i * 100) for i in range(n)],
            "channel_id": "ch_01",
            "current_a": rng.normal(5.0, 0.3, n),
            "voltage_v": 13.5,
            "temperature_c": 40.0 + rng.normal(0, 1, n),
            "state_on_off": state_on,
            "trip_flag": False,
            "overload_flag": False,
            "reset_counter": 0,
            "pwm_duty_pct": 100.0,
            "anomaly_score": 0.0,
            "is_anomaly": False,
            "predicted_fault": "none",
            "fault_confidence": 0.0,
        }
    )
    # Inject anomalies
    n_anom = int(n * anomaly_frac)
    if n_anom > 0:
        idxs = rng.choice(n, n_anom, replace=False)
        df.loc[idxs, "is_anomaly"] = True
        df.loc[idxs, "anomaly_score"] = rng.uniform(0.6, 1.0, n_anom)

    # Inject trips
    n_trip = int(n * trip_frac)
    if n_trip > 0:
        trip_idxs = rng.choice(n, n_trip, replace=False)
        df.loc[trip_idxs, "trip_flag"] = True

    return df


def _boundary_df(
    on_rows: int = 50,
    off_rows: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a DataFrame with an on-period followed by off-period."""
    on_part = _scored_df(on_rows, state_on=True, seed=seed)
    off_part = _scored_df(off_rows, state_on=False, seed=seed + 1)
    # Shift off timestamps to follow on
    off_start = on_part["timestamp"].iloc[-1] + timedelta(milliseconds=100)
    off_part["timestamp"] = [
        off_start + timedelta(milliseconds=i * 100) for i in range(off_rows)
    ]
    return pd.concat([on_part, off_part], ignore_index=True)


# ---------------------------------------------------------------------------
# CycleSummary schema tests
# ---------------------------------------------------------------------------


class TestCycleSummarySchema:
    def test_defaults(self):
        s = CycleSummary(
            cycle_id="abc123",
            open_timestamp=T0,
            close_timestamp=T0 + timedelta(seconds=60),
        )
        assert s.health_band == HealthBand.NOMINAL
        assert s.dominant_fault == FaultType.NONE
        assert s.anomaly_count == 0
        assert s.duration_s == 0.0  # caller must set explicitly

    def test_serialization_round_trip(self):
        s = CycleSummary(
            cycle_id="xyz",
            open_timestamp=T0,
            close_timestamp=T0 + timedelta(seconds=30),
            duration_s=30.0,
            anomaly_count=5,
            health_band=HealthBand.DEGRADED,
        )
        d = s.model_dump()
        assert d["health_band"] == "degraded"
        assert d["cycle_id"] == "xyz"
        s2 = CycleSummary.model_validate(d)
        assert s2.anomaly_count == 5


class TestHealthBand:
    def test_values(self):
        assert HealthBand.NOMINAL.value == "nominal"
        assert HealthBand.CRITICAL.value == "critical"


# ---------------------------------------------------------------------------
# CycleAccumulator unit tests
# ---------------------------------------------------------------------------


class TestCycleAccumulatorManual:
    """Test manual open/close (no boundary detection)."""

    def test_open_close_produces_summary(self):
        acc = CycleAccumulator(boundary_column=None)
        acc.open(T0)
        assert acc.is_open

        scored = _scored_df(50, anomaly_frac=0.1)
        acc.ingest(scored)

        s = acc.close(T0 + timedelta(seconds=5))
        assert s is not None
        assert not acc.is_open
        assert s.sample_count == 50
        assert s.anomaly_count > 0
        assert s.duration_s == 5.0

    def test_close_without_open_returns_none(self):
        acc = CycleAccumulator(boundary_column=None)
        assert acc.close() is None

    def test_counters_reset_between_cycles(self):
        acc = CycleAccumulator(boundary_column=None)

        # Cycle 1 with anomalies
        acc.open(T0)
        acc.ingest(_scored_df(20, anomaly_frac=0.5))
        s1 = acc.close(T0 + timedelta(seconds=2))

        # Cycle 2 clean
        acc.open(T0 + timedelta(seconds=3))
        acc.ingest(_scored_df(20, anomaly_frac=0.0))
        s2 = acc.close(T0 + timedelta(seconds=5))

        assert s1.anomaly_count > 0
        assert s2.anomaly_count == 0
        assert len(acc.completed) == 2

    def test_peak_tracking(self):
        acc = CycleAccumulator(boundary_column=None)
        acc.open(T0)

        df = _scored_df(10)
        df.loc[3, "current_a"] = 99.0
        df.loc[7, "temperature_c"] = 150.0
        acc.ingest(df)

        s = acc.close(T0 + timedelta(seconds=1))
        assert s.peak_current_a == 99.0
        assert s.peak_temperature_c == 150.0

    def test_trip_counting(self):
        acc = CycleAccumulator(boundary_column=None)
        acc.open(T0)
        acc.ingest(_scored_df(100, trip_frac=0.05))
        s = acc.close(T0 + timedelta(seconds=10))
        assert s.trip_count == 5  # 5% of 100

    def test_high_temp_dwell(self):
        acc = CycleAccumulator(
            boundary_column=None,
            high_temp_threshold_c=50.0,
        )
        acc.open(T0)
        df = _scored_df(10)
        df["temperature_c"] = 60.0  # all above threshold
        acc.ingest(df, sample_interval_s=1.0)
        s = acc.close(T0 + timedelta(seconds=10))
        assert s.high_temp_dwell_s == pytest.approx(10.0)

    def test_voltage_sag_dwell(self):
        acc = CycleAccumulator(
            boundary_column=None,
            low_voltage_threshold_v=11.0,
        )
        acc.open(T0)
        df = _scored_df(20)
        df.loc[5:14, "voltage_v"] = 9.0  # 10 rows below threshold
        acc.ingest(df, sample_interval_s=0.1)
        s = acc.close(T0 + timedelta(seconds=2))
        assert s.voltage_sag_dwell_s == pytest.approx(1.0)


class TestCycleAccumulatorBoundary:
    """Test automatic boundary detection via state_on_off."""

    def test_falling_edge_closes_cycle(self):
        acc = CycleAccumulator(boundary_column="state_on_off")
        df = _boundary_df(on_rows=50, off_rows=10)
        summaries = acc.ingest(df)
        # Falling edge should produce one summary
        assert len(summaries) == 1
        assert summaries[0].sample_count == 50
        assert not acc.is_open

    def test_multiple_cycles(self):
        acc = CycleAccumulator(boundary_column="state_on_off")

        # Cycle 1: on then off
        df1 = _boundary_df(on_rows=30, off_rows=5, seed=1)
        s1 = acc.ingest(df1)

        # Cycle 2: on then off
        df2 = _boundary_df(on_rows=40, off_rows=5, seed=2)
        # Shift timestamps
        last_ts = df1["timestamp"].iloc[-1]
        offset = last_ts + timedelta(milliseconds=100) - df2["timestamp"].iloc[0]
        df2["timestamp"] = df2["timestamp"] + offset
        s2 = acc.ingest(df2)

        assert len(s1) == 1
        assert len(s2) == 1
        assert s1[0].sample_count == 30
        assert s2[0].sample_count == 40
        assert len(acc.completed) == 2

    def test_auto_open_without_boundary_column(self):
        acc = CycleAccumulator(boundary_column=None)
        df = _scored_df(20)
        sums = acc.ingest(df)
        # No boundary → auto-opens, never closes
        assert len(sums) == 0
        assert acc.is_open
        assert acc._sample_count == 20


class TestCycleAccumulatorStress:
    """Test stress scoring and health bands."""

    def test_clean_cycle_is_nominal(self):
        acc = CycleAccumulator(boundary_column=None)
        acc.open(T0)
        acc.ingest(_scored_df(100, anomaly_frac=0.0, trip_frac=0.0))
        s = acc.close(T0 + timedelta(seconds=10))
        assert s.health_band == HealthBand.NOMINAL
        assert s.cycle_stress < 0.15

    def test_stressed_cycle_is_degraded_or_critical(self):
        acc = CycleAccumulator(
            boundary_column=None,
            high_temp_threshold_c=30.0,  # low → most rows trigger
        )
        acc.open(T0)
        acc.ingest(_scored_df(100, anomaly_frac=0.5, trip_frac=0.1))
        s = acc.close(T0 + timedelta(seconds=10))
        assert s.health_band in (HealthBand.DEGRADED, HealthBand.CRITICAL)
        assert s.cycle_stress > 0.3

    def test_dominant_fault_tracking(self):
        acc = CycleAccumulator(boundary_column=None)
        acc.open(T0)

        df = _scored_df(50)
        # 30 rows with overload_spike, 10 with thermal_drift
        df.loc[:29, "predicted_fault"] = "overload_spike"
        df.loc[:29, "fault_confidence"] = 0.8
        df.loc[30:39, "predicted_fault"] = "thermal_drift"
        df.loc[30:39, "fault_confidence"] = 0.7
        acc.ingest(df)

        s = acc.close(T0 + timedelta(seconds=5))
        assert s.dominant_fault == FaultType.OVERLOAD_SPIKE
        assert s.dominant_fault_confidence > 0.0


# ---------------------------------------------------------------------------
# Integration with EdgeRuntime
# ---------------------------------------------------------------------------


class TestEdgeRuntimeCycleIntegration:
    def _make_telemetry(self, n: int = 200, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "timestamp": [
                    T0 + timedelta(milliseconds=i * 100) for i in range(n)
                ],
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

    def test_cycle_tracking_disabled_by_default(self):
        df = self._make_telemetry(100)
        transport = DataFrameTransport(df)
        pipeline = InferencePipeline()
        cfg = EdgeConfig(batch_size=25)
        runtime = EdgeRuntime(transport, pipeline, cfg)
        assert runtime._cycle_accumulator is None
        runtime.run()
        assert len(runtime.cycle_summaries) == 0

    def test_cycle_tracking_enabled_produces_summary(self):
        df = self._make_telemetry(200)
        transport = DataFrameTransport(df)
        pipeline = InferencePipeline()
        cfg = EdgeConfig(
            batch_size=50,
            cycle_tracking_enabled=True,
            cycle_boundary_column=None,  # no boundary → auto-open, close at end
        )
        runtime = EdgeRuntime(transport, pipeline, cfg)
        runtime.run()
        # Should have at least one summary (created in finally block)
        assert len(runtime.cycle_summaries) >= 1
        s = runtime.cycle_summaries[0]
        assert s.sample_count > 0
        assert s.cycle_type == "ignition"

    def test_cycle_tracking_with_boundary(self):
        # Create data: 100 rows on, 20 rows off, 80 rows on
        rng = np.random.default_rng(99)
        rows = []
        for i in range(200):
            on = not (100 <= i < 120)  # off in middle
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
            cycle_boundary_column="state_on_off",
        )
        runtime = EdgeRuntime(transport, pipeline, cfg)
        runtime.run()
        # At least one completed cycle from the off transition, plus one from finally
        assert len(runtime.cycle_summaries) >= 1
