"""Cycle tracking — accumulate per-cycle counters and produce compact summaries.

The CycleAccumulator is fed scored DataFrames from the EdgeRuntime on every
iteration.  It maintains lightweight RAM-only counters while a cycle is open.
When a cycle boundary is detected (or forced via ``close()``), it produces
a :class:`CycleSummary` and resets.

Cycle boundaries are detected by a configurable signal column (default:
``state_on_off``).  A cycle opens when the signal transitions to *True* and
closes when it transitions to *False*.  If no boundary signal is available
the caller can manage open/close explicitly.
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from src.schemas.telemetry import CycleSummary, FaultType, HealthBand
from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration (kept minimal — lives next to the accumulator, not in YAML)
# ---------------------------------------------------------------------------

_DEFAULT_HIGH_CURRENT_FACTOR = 0.8  # fraction of fuse_rating → "high load"
_DEFAULT_HIGH_TEMP_C = 125.0  # junction temp threshold for "high temp"
_DEFAULT_LOW_VOLTAGE_V = 10.0  # below this → voltage sag


class CycleAccumulator:
    """RAM-only accumulator for one open cycle.

    Parameters
    ----------
    cycle_type:
        Label for the kind of cycle (``"ignition"``, ``"drive"``, ``"charge"``).
    boundary_column:
        DataFrame column whose True→False edge signals cycle-close.
        Set to ``None`` to disable automatic boundary detection.
    high_current_threshold_a:
        Absolute current above which dwell time is counted.  Overrides the
        per-channel ratio when set explicitly.
    high_temp_threshold_c:
        Junction temperature above which dwell time is counted.
    low_voltage_threshold_v:
        Supply voltage below which sag dwell is counted.
    """

    def __init__(
        self,
        cycle_type: str = "ignition",
        boundary_column: str | None = "state_on_off",
        high_current_threshold_a: float | None = None,
        high_temp_threshold_c: float = _DEFAULT_HIGH_TEMP_C,
        low_voltage_threshold_v: float = _DEFAULT_LOW_VOLTAGE_V,
    ) -> None:
        self.cycle_type = cycle_type
        self.boundary_column = boundary_column
        self.high_current_threshold_a = high_current_threshold_a
        self.high_temp_threshold_c = high_temp_threshold_c
        self.low_voltage_threshold_v = low_voltage_threshold_v

        # State
        self._open = False
        self._cycle_id: str = ""
        self._open_ts: datetime | None = None
        self._last_boundary_value: bool | None = None

        # Counters (reset each cycle)
        self._anomaly_count = 0
        self._trip_count = 0
        self._retry_count = 0
        self._sample_count = 0
        self._peak_current = 0.0
        self._peak_temperature = 0.0
        self._high_load_dwell_s = 0.0
        self._high_temp_dwell_s = 0.0
        self._voltage_sag_dwell_s = 0.0
        self._fault_votes: Counter[str] = Counter()

        # Completed summaries
        self.completed: list[CycleSummary] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self, timestamp: datetime | None = None) -> None:
        """Explicitly open a new cycle."""
        if self._open:
            log.warning("Cycle already open — closing previous before re-opening")
            self.close(timestamp)
        self._cycle_id = uuid.uuid4().hex[:12]
        self._open_ts = timestamp or datetime.now(tz=timezone.utc)
        self._open = True
        self._reset_counters()
        log.debug("Cycle opened: %s at %s", self._cycle_id, self._open_ts)

    def close(self, timestamp: datetime | None = None) -> CycleSummary | None:
        """Close the current cycle, produce a summary, and reset.

        Returns ``None`` if no cycle was open.
        """
        if not self._open:
            return None

        close_ts = timestamp or datetime.now(tz=timezone.utc)
        duration = (close_ts - self._open_ts).total_seconds() if self._open_ts else 0.0

        dominant_fault, dominant_conf = self._dominant_fault()
        stress = self._compute_stress()
        band = self._stress_to_band(stress)

        summary = CycleSummary(
            cycle_id=self._cycle_id,
            cycle_type=self.cycle_type,
            open_timestamp=self._open_ts or close_ts,
            close_timestamp=close_ts,
            duration_s=round(duration, 2),
            anomaly_count=self._anomaly_count,
            trip_count=self._trip_count,
            retry_count=self._retry_count,
            sample_count=self._sample_count,
            peak_current_a=round(self._peak_current, 3),
            peak_temperature_c=round(self._peak_temperature, 2),
            high_load_dwell_s=round(self._high_load_dwell_s, 2),
            high_temp_dwell_s=round(self._high_temp_dwell_s, 2),
            voltage_sag_dwell_s=round(self._voltage_sag_dwell_s, 2),
            dominant_fault=dominant_fault,
            dominant_fault_confidence=round(dominant_conf, 3),
            cycle_stress=round(stress, 3),
            health_band=band,
        )

        self.completed.append(summary)
        self._open = False
        log.info(
            "Cycle closed: %s  stress=%.2f  band=%s  anomalies=%d  trips=%d",
            self._cycle_id,
            stress,
            band.value,
            self._anomaly_count,
            self._trip_count,
        )
        return summary

    def ingest(self, scored: pd.DataFrame, sample_interval_s: float = 0.1) -> list[CycleSummary]:
        """Feed a scored DataFrame from the inference pipeline.

        Detects cycle boundaries automatically (if ``boundary_column`` is set),
        accumulates counters, and returns any summaries produced during this
        batch (usually zero or one).

        Parameters
        ----------
        scored:
            DataFrame with at least ``is_anomaly``, ``anomaly_score``,
            ``predicted_fault``, ``fault_confidence`` columns.
        sample_interval_s:
            Approximate time between consecutive rows — used for dwell
            estimation.

        Returns
        -------
        List of CycleSummary produced during this batch (may be empty).
        """
        new_summaries: list[CycleSummary] = []

        # Detect boundaries and auto-open/close
        if self.boundary_column and self.boundary_column in scored.columns:
            for idx, row in scored.iterrows():
                val = bool(row[self.boundary_column])
                prev = self._last_boundary_value
                self._last_boundary_value = val

                # Rising edge → open
                if val and (prev is None or not prev):
                    ts = row.get("timestamp", datetime.now(tz=timezone.utc))
                    if self._open:
                        s = self.close(ts)
                        if s:
                            new_summaries.append(s)
                    self.open(ts)

                # Falling edge → close
                elif not val and prev:
                    ts = row.get("timestamp", datetime.now(tz=timezone.utc))
                    s = self.close(ts)
                    if s:
                        new_summaries.append(s)

                # Accumulate if open
                if self._open:
                    self._accumulate_row(row, sample_interval_s)
        else:
            # No boundary column — auto-open on first data if not open
            if not self._open:
                first_ts = scored["timestamp"].iloc[0] if "timestamp" in scored.columns else None
                self.open(first_ts)

            for _, row in scored.iterrows():
                self._accumulate_row(row, sample_interval_s)

        return new_summaries

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _accumulate_row(self, row: pd.Series, interval_s: float) -> None:
        """Update counters from one scored row."""
        self._sample_count += 1

        # Anomaly
        if row.get("is_anomaly", False):
            self._anomaly_count += 1

        # Trips and retries
        if row.get("trip_flag", False):
            self._trip_count += 1
        reset_ctr = row.get("reset_counter", 0)
        if isinstance(reset_ctr, (int, float)) and reset_ctr > 0:
            self._retry_count = max(self._retry_count, int(reset_ctr))

        # Peaks
        current = row.get("current_a", 0.0)
        if isinstance(current, (int, float)):
            self._peak_current = max(self._peak_current, float(current))

        temp = row.get("temperature_c", 0.0)
        if isinstance(temp, (int, float)):
            self._peak_temperature = max(self._peak_temperature, float(temp))

        # Dwell times
        if self.high_current_threshold_a is not None and isinstance(current, (int, float)):
            if float(current) > self.high_current_threshold_a:
                self._high_load_dwell_s += interval_s

        if isinstance(temp, (int, float)) and float(temp) > self.high_temp_threshold_c:
            self._high_temp_dwell_s += interval_s

        voltage = row.get("voltage_v", 14.0)
        if isinstance(voltage, (int, float)) and float(voltage) < self.low_voltage_threshold_v:
            self._voltage_sag_dwell_s += interval_s

        # Fault votes
        fault = row.get("predicted_fault", "none")
        confidence = float(row.get("fault_confidence", 0.0))
        if fault and str(fault) != "none" and confidence > 0:
            self._fault_votes[str(fault)] += 1

    def _dominant_fault(self) -> tuple[FaultType, float]:
        """Return the most-voted fault and a rough confidence."""
        if not self._fault_votes:
            return FaultType.NONE, 0.0
        top_fault_str, count = self._fault_votes.most_common(1)[0]
        total_faults = sum(self._fault_votes.values())
        confidence = count / total_faults if total_faults > 0 else 0.0
        try:
            fault = FaultType(top_fault_str)
        except ValueError:
            fault = FaultType.NONE
        return fault, confidence

    def _compute_stress(self) -> float:
        """Heuristic cycle-stress score in [0, 1].

        Combines anomaly burden, trip burden, and thermal burden with
        simple clamped weights.  Intentionally kept dead-simple so it
        can be tuned later via config.
        """
        n = max(self._sample_count, 1)

        anomaly_rate = min(self._anomaly_count / n, 1.0)
        trip_burden = min(self._trip_count / 10.0, 1.0)  # 10 trips → max
        retry_burden = min(self._retry_count / 5.0, 1.0)

        # Thermal: proportion of cycle in high-temp zone
        if self._open_ts:
            close_ts = datetime.now(tz=timezone.utc)
            duration = max((close_ts - self._open_ts).total_seconds(), 1.0)
        else:
            duration = max(self._sample_count * 0.1, 1.0)
        thermal_burden = min(self._high_temp_dwell_s / duration, 1.0)

        # Weighted sum (tunable later)
        stress = (
            0.35 * anomaly_rate
            + 0.25 * trip_burden
            + 0.15 * retry_burden
            + 0.25 * thermal_burden
        )
        return min(stress, 1.0)

    @staticmethod
    def _stress_to_band(stress: float) -> HealthBand:
        if stress < 0.15:
            return HealthBand.NOMINAL
        if stress < 0.40:
            return HealthBand.MONITOR
        if stress < 0.70:
            return HealthBand.DEGRADED
        return HealthBand.CRITICAL

    def _reset_counters(self) -> None:
        self._anomaly_count = 0
        self._trip_count = 0
        self._retry_count = 0
        self._sample_count = 0
        self._peak_current = 0.0
        self._peak_temperature = 0.0
        self._high_load_dwell_s = 0.0
        self._high_temp_dwell_s = 0.0
        self._voltage_sag_dwell_s = 0.0
        self._fault_votes.clear()
