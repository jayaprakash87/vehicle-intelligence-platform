"""Ingestion — parse, validate, normalize raw telemetry rows.

Sits between transport and feature engine.  Handles missing values,
type coercion, and schema validation for rows coming off the bus.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.schemas.telemetry import TelemetryRecord
from src.utils.logging import get_logger

log = get_logger(__name__)


class Normalizer:
    """Cleans and normalizes raw telemetry DataFrames."""

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard cleaning to a raw telemetry DataFrame."""
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Sort by time then channel
        df.sort_values(["timestamp", "channel_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Track missing data rate per channel BEFORE forward-filling
        # This preserves evidence of dropped packets for downstream detection
        if "current_a" in df.columns:
            df["_is_missing"] = df["current_a"].isna().astype(float)
            df["missing_rate"] = df.groupby("channel_id")["_is_missing"].transform(
                lambda s: s.rolling(50, min_periods=1).mean()
            )
            df.drop(columns=["_is_missing"], inplace=True)
        else:
            df["missing_rate"] = 0.0

        # Forward-fill NaN in current/voltage (dropped-packet scenario)
        # Limit to 5 rows — beyond that, data is stale rather than filled
        for col in ("current_a", "voltage_v"):
            if col in df.columns:
                df[col] = df.groupby("channel_id")[col].ffill(limit=5)

        # Track forward-fill staleness: rows where NaN persists after limited ffill
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

        return df

    def validate_record(self, row: dict) -> Optional[TelemetryRecord]:
        """Validate a single row dict against the schema. Returns None on failure."""
        try:
            return TelemetryRecord.model_validate(row)
        except Exception as exc:
            log.debug("Validation failed for row: %s", exc)
            return None
