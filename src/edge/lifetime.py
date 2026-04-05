"""Lifetime health tracking — histogram-based load spectra with derived health.

The :class:`LifetimeHealthTracker` is fed completed :class:`CycleSummary`
objects and maintains :class:`LifetimeHealthState` in RAM.  The primary
state is six fixed-bin histograms (load spectra) — the automotive-standard
approach for compact on-vehicle lifetime tracking.

Design principles:
  - Histograms are the source of truth (≈192 bytes total, NvM-friendly)
  - Health score + trend are **derived** from histogram shape on each update
  - Distribution shift toward upper bins = degradation signal
  - Maps to reliability models (Miner's rule, Wöhler curves)
"""

from __future__ import annotations

from src.schemas.telemetry import (
    CycleSummary,
    HealthBand,
    LifetimeHealthState,
    TrendDirection,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


class LifetimeHealthTracker:
    """RAM-only tracker updated once per completed cycle.

    Parameters
    ----------
    trend_window:
        Number of recent health scores to keep for trend detection.
    upper_bins:
        How many of the highest bins count as "stressed" when computing
        the upper-tail fraction for each histogram.
    """

    def __init__(
        self,
        trend_window: int = 5,
        upper_bins: int = 2,
    ) -> None:
        self.trend_window = max(3, trend_window)
        self.upper_bins = max(1, upper_bins)

        self._state = LifetimeHealthState()
        self._recent_scores: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> LifetimeHealthState:
        """Current lifetime health state (read-only snapshot)."""
        return self._state.model_copy(deep=True)

    def ingest(self, summary: CycleSummary) -> LifetimeHealthState:
        """Update lifetime state from a completed cycle summary.

        1. Record values into histograms (primary state).
        2. Derive health score from histogram upper-tail fractions.
        3. Detect trend from recent score history.
        """
        s = self._state

        # --- 1. Record into histograms ---
        s.peak_current_hist.record(summary.peak_current_a)
        s.peak_temperature_hist.record(summary.peak_temperature_c)
        s.cycle_stress_hist.record(summary.cycle_stress)
        s.trips_per_cycle_hist.record(float(summary.trip_count))
        s.retries_per_cycle_hist.record(float(summary.retry_count))

        # Thermal dwell fraction
        duration = max(summary.duration_s, 1.0)
        thermal_frac = min(summary.high_temp_dwell_s / duration, 1.0)
        s.thermal_dwell_frac_hist.record(thermal_frac)

        # --- 2. Derive health score from histogram shape ---
        ub = self.upper_bins
        stress_fractions = [
            s.peak_current_hist.upper_fraction(ub),
            s.peak_temperature_hist.upper_fraction(ub),
            s.cycle_stress_hist.upper_fraction(ub),
            s.trips_per_cycle_hist.upper_fraction(ub),
            s.retries_per_cycle_hist.upper_fraction(ub),
            s.thermal_dwell_frac_hist.upper_fraction(ub),
        ]
        # Weighted combination — same weight philosophy as cycle stress
        weights = [0.20, 0.20, 0.20, 0.15, 0.10, 0.15]
        weighted_upper = sum(w * f for w, f in zip(weights, stress_fractions))

        s.health_score = round(max(1.0 - weighted_upper, 0.0), 4)
        s.health_band = self._score_to_band(s.health_score)

        # --- 3. Trend detection ---
        self._recent_scores.append(s.health_score)
        if len(self._recent_scores) > self.trend_window:
            self._recent_scores = self._recent_scores[-self.trend_window :]
        s.trend = self._compute_trend()

        # --- Bookkeeping ---
        s.cycles_ingested += 1
        s.last_cycle_id = summary.cycle_id
        s.last_update_timestamp = summary.close_timestamp

        log.info(
            "Lifetime updated: cycle=%s  health=%.3f  band=%s  trend=%s  cycles=%d",
            summary.cycle_id,
            s.health_score,
            s.health_band.value,
            s.trend.value,
            s.cycles_ingested,
        )
        return self.state

    def reset(self) -> None:
        """Reset to initial state."""
        self._state = LifetimeHealthState()
        self._recent_scores.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_band(score: float) -> HealthBand:
        """Map health score (1=healthy, 0=critical) to HealthBand."""
        if score > 0.85:
            return HealthBand.NOMINAL
        if score > 0.60:
            return HealthBand.MONITOR
        if score > 0.30:
            return HealthBand.DEGRADED
        return HealthBand.CRITICAL

    def _compute_trend(self) -> TrendDirection:
        """Simple linear trend over recent health scores."""
        scores = self._recent_scores
        n = len(scores)
        if n < 3:
            return TrendDirection.STABLE

        # Compare first half avg to second half avg
        mid = n // 2
        first_half = sum(scores[:mid]) / mid
        second_half = sum(scores[mid:]) / (n - mid)
        delta = second_half - first_half

        if delta > 0.05:
            return TrendDirection.IMPROVING
        if delta < -0.10:
            return TrendDirection.WORSENING
        if delta < -0.03:
            return TrendDirection.DEGRADING
        return TrendDirection.STABLE

    def _compute_trend(self) -> TrendDirection:
        """Simple linear trend over recent health scores."""
        scores = self._recent_scores
        n = len(scores)
        if n < 3:
            return TrendDirection.STABLE

        # Simple slope: compare first half avg to second half avg
        mid = n // 2
        first_half = sum(scores[:mid]) / mid
        second_half = sum(scores[mid:]) / (n - mid)
        delta = second_half - first_half

        if delta > 0.05:
            return TrendDirection.IMPROVING
        if delta < -0.10:
            return TrendDirection.WORSENING
        if delta < -0.03:
            return TrendDirection.DEGRADING
        return TrendDirection.STABLE
