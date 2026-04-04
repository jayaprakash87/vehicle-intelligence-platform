"""Ingestion — parse, validate, normalize raw telemetry rows.

Sits between transport and feature engine.  Handles missing values,
type coercion, schema validation, and multi-rate resampling for rows
coming off CAN or XCP.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config.models import NormalizerConfig
from src.schemas.telemetry import TelemetryRecord
from src.utils.logging import get_logger

log = get_logger(__name__)


class Normalizer:
    """Cleans and normalizes raw telemetry DataFrames.

    Supports multi-rate channels (e.g. XCP 10ms + 50ms rasters) by:
    1. Computing per-channel sample intervals
    2. Applying time-based forward-fill limits
    3. Optional resampling to a common grid
    """

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.cfg = config or NormalizerConfig()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard cleaning to a raw telemetry DataFrame."""
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Sort by time then channel
        df.sort_values(["timestamp", "channel_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Compute per-channel sample intervals (for time-based limits)
        ch_intervals = self._estimate_channel_intervals(df)

        # Track missing data rate per channel BEFORE forward-filling
        if "current_a" in df.columns:
            df["_is_missing"] = df["current_a"].isna().astype(float)
            for ch_id, interval_s in ch_intervals.items():
                mask = df["channel_id"] == ch_id
                win = max(int(self.cfg.missing_rate_window_s / interval_s), 1)
                df.loc[mask, "missing_rate"] = (
                    df.loc[mask, "_is_missing"].rolling(win, min_periods=1).mean().values
                )
            if "missing_rate" not in df.columns:
                df["missing_rate"] = 0.0
            df.drop(columns=["_is_missing"], inplace=True)
        else:
            df["missing_rate"] = 0.0

        # Forward-fill NaN — time-based limit per channel
        for col in ("current_a", "voltage_v"):
            if col in df.columns:
                for ch_id, interval_s in ch_intervals.items():
                    limit = max(int(self.cfg.ffill_tolerance_s / interval_s), 1)
                    mask = df["channel_id"] == ch_id
                    df.loc[mask, col] = df.loc[mask, col].ffill(limit=limit)

        # Track forward-fill staleness
        for col in ("current_a", "voltage_v"):
            if col in df.columns:
                df[f"_stale_{col}"] = df[col].isna()

        # Clip obviously out-of-range values
        df["current_a"] = df["current_a"].clip(-1, 200)
        df["voltage_v"] = df["voltage_v"].clip(0, 60)
        df["temperature_c"] = df["temperature_c"].clip(-40, 150)
        df["pwm_duty_pct"] = df["pwm_duty_pct"].clip(0, 100)

        # Ensure boolean columns
        for col in ("state_on_off", "trip_flag", "overload_flag"):
            if col in df.columns:
                df[col] = df[col].astype(bool)

        n_null = df[["current_a", "voltage_v"]].isna().sum().sum()
        if n_null > 0:
            log.warning("After normalization, %d null signal values remain", n_null)

        # Optional resampling to a common grid
        if self.cfg.resample_interval_ms > 0:
            df = self._resample(df)

        return df

    def validate_record(self, row: dict) -> Optional[TelemetryRecord]:
        """Validate a single row dict against the schema. Returns None on failure."""
        try:
            return TelemetryRecord.model_validate(row)
        except Exception as exc:
            log.debug("Validation failed for row: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Multi-rate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_channel_intervals(df: pd.DataFrame) -> dict[str, float]:
        """Return {channel_id: interval_seconds} estimated from timestamps."""
        intervals: dict[str, float] = {}
        for ch_id in df["channel_id"].unique():
            ch_df = df[df["channel_id"] == ch_id]
            if len(ch_df) >= 2:
                ts = ch_df["timestamp"]
                median_dt = ts.diff().dropna().dt.total_seconds().median()
                intervals[ch_id] = max(median_dt, 0.001)
            else:
                intervals[ch_id] = 0.1  # fallback
        return intervals

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample all channels to a common time grid via nearest-neighbor.

        This aligns multi-rate XCP rasters (e.g. 10ms current + 50ms temp)
        onto a single grid for downstream feature computation.
        """
        target_ms = self.cfg.resample_interval_ms
        if target_ms <= 0:
            return df

        t_min = df["timestamp"].min()
        t_max = df["timestamp"].max()
        freq = pd.Timedelta(milliseconds=target_ms)
        common_grid = pd.date_range(start=t_min, end=t_max, freq=freq)

        resampled_parts = []
        for ch_id, ch_df in df.groupby("channel_id"):
            ch_df = ch_df.set_index("timestamp").sort_index()
            # Reindex to common grid — forward-fill for slow signals, then back-fill start
            ch_re = ch_df.reindex(common_grid, method="nearest", tolerance=freq * 2)
            ch_re["timestamp"] = common_grid
            ch_re["channel_id"] = ch_id
            # Ensure boolean columns don't have NaN (fill with False)
            for col in ("state_on_off", "trip_flag", "overload_flag"):
                if col in ch_re.columns:
                    ch_re[col] = ch_re[col].fillna(False).astype(bool)
            ch_re.reset_index(drop=True, inplace=True)
            resampled_parts.append(ch_re)

        result = pd.concat(resampled_parts, ignore_index=True)
        result.sort_values(["timestamp", "channel_id"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        log.info(
            "Resampled %d rows to %dms grid → %d rows",
            len(df),
            int(target_ms),
            len(result),
        )
        return result
