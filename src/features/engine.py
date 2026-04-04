"""Feature engineering — rolling / derived metrics from normalized telemetry.

Operates on per-channel DataFrames.  Designed to work in batch (full DF)
or incrementally (append new rows and recompute tail).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.models import FeatureConfig
from src.utils.logging import get_logger

log = get_logger(__name__)


class FeatureEngine:
    """Computes derived features from normalized telemetry."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.cfg = config or FeatureConfig()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived feature columns to a normalized telemetry DataFrame.

        The input must be sorted by (channel_id, timestamp).
        Returns a new DataFrame with feature columns appended.
        """
        df = df.copy()
        df.sort_values(["channel_id", "timestamp"], inplace=True)

        w = self.cfg.window_size
        mp = self.cfg.min_periods

        grouped = df.groupby("channel_id")

        # Rolling current statistics
        df["rolling_rms_current"] = grouped["current_a"].transform(
            lambda s: (s ** 2).rolling(w, min_periods=mp).mean().pipe(np.sqrt)
        )
        df["rolling_mean_current"] = grouped["current_a"].transform(
            lambda s: s.rolling(w, min_periods=mp).mean()
        )
        df["rolling_max_current"] = grouped["current_a"].transform(
            lambda s: s.rolling(w, min_periods=mp).max()
        )
        df["rolling_min_current"] = grouped["current_a"].transform(
            lambda s: s.rolling(w, min_periods=mp).min()
        )

        # Temperature slope (finite difference over window)
        df["temperature_slope"] = grouped["temperature_c"].transform(
            lambda s: s.diff(periods=w).fillna(0) / max(w, 1)
        )

        # Spike score — how many σ above rolling mean
        roll_mean = grouped["current_a"].transform(
            lambda s: s.rolling(w, min_periods=mp).mean()
        )
        roll_std = grouped["current_a"].transform(
            lambda s: s.rolling(w, min_periods=mp).std()
        )
        df["spike_score"] = ((df["current_a"] - roll_mean) / roll_std.replace(0, 1)).clip(lower=0)

        # Trip frequency — rolling sum of trip_flag transitions
        df["_trip_edge"] = grouped["trip_flag"].transform(
            lambda s: s.astype(int).diff().clip(lower=0)
        )
        df["trip_frequency"] = grouped["_trip_edge"].transform(
            lambda s: s.rolling(w * 2, min_periods=1).sum()
        )
        df.drop(columns=["_trip_edge"], inplace=True)

        # Recovery time — vectorized: seconds since last trip falling edge
        interval = self._interval_s(df)
        df["recovery_time_s"] = grouped["trip_flag"].transform(
            lambda s: self._vectorized_recovery(s, interval)
        )

        # Degradation trend — slope of rolling_mean_current over long window
        long_w = w * 4
        df["degradation_trend"] = grouped["rolling_mean_current"].transform(
            lambda s: s.rolling(long_w, min_periods=w).apply(
                self._linear_slope, raw=True
            )
        ).fillna(0)

        # Anomaly score placeholder (filled by model layer later)
        if "anomaly_score" not in df.columns:
            df["anomaly_score"] = 0.0

        log.info("Computed features for %d rows", len(df))
        return df

    # ------------------------------------------------------------------

    @staticmethod
    def _linear_slope(arr: np.ndarray) -> float:
        """Least-squares slope of an array treated as evenly spaced."""
        n = len(arr)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        slope = (n * np.dot(x, arr) - x.sum() * arr.sum()) / (n * np.dot(x, x) - x.sum() ** 2 + 1e-12)
        return float(slope)

    @staticmethod
    def _vectorized_recovery(trip_series: pd.Series, interval_s: float) -> pd.Series:
        """Compute recovery_time_s vectorized for a single channel's trip_flag series.

        Recovery time counts up from 0 at each trip falling edge (True → False),
        continues incrementing while trip_flag is False, and resets to 0 when
        trip_flag goes True again or at the next falling edge.
        """
        trip = trip_series.astype(int)
        # Falling edge: trip goes from 1 to 0
        falling = (trip.diff() == -1)
        # We only count recovery while trip is False after a falling edge
        # Group by cumulative falling edges to segment recovery windows
        recovery_group = falling.cumsum()
        # Only count in non-trip regions
        not_tripped = ~trip_series.astype(bool)
        # Within each recovery group, count rows since group start
        row_count = recovery_group.groupby(recovery_group).cumcount()
        # Recovery time = row count * interval, but only where not tripped and after at least one trip end
        result = pd.Series(0.0, index=trip_series.index)
        active = not_tripped & (recovery_group > 0)
        result[active] = row_count[active] * interval_s
        return result

    @staticmethod
    def _interval_s(df: pd.DataFrame) -> float:
        """Estimate sample interval in seconds from timestamps within a single channel."""
        if len(df) < 2:
            return 0.1
        # Use the first channel's consecutive timestamps to avoid cross-channel errors
        if "channel_id" in df.columns:
            first_ch = df["channel_id"].iloc[0]
            ch_df = df[df["channel_id"] == first_ch]
            if len(ch_df) >= 2:
                dt = (ch_df["timestamp"].iloc[1] - ch_df["timestamp"].iloc[0]).total_seconds()
                return max(dt, 0.001)
        dt = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()
        return max(dt, 0.001)
