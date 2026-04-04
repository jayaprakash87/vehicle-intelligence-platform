"""Measurement file reader — loads real data from MDF4, CSV, or Parquet.

Supports ASAM MDF4 (the automotive standard for bench/in-vehicle recordings),
plus CSV and Parquet for pre-exported data.  A configurable column mapping
translates OEM-specific signal names to VIP's canonical schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.schemas.telemetry import DeviceStatus, ProtectionEvent
from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Column mapping config
# ---------------------------------------------------------------------------

class ColumnMapping(BaseModel):
    """Maps OEM / test-bench signal names to VIP canonical columns.

    Keys are VIP column names; values are source signal names
    (as they appear in the MDF4 / CSV file).  A value of "" means
    the column is absent and should be filled with a default.
    """

    timestamp: str = Field(default="timestamp", description="Timestamp column or MDF4 time channel")
    channel_id: str = Field(default="channel_id", description="Channel identifier column")
    current_a: str = Field(default="current_a", description="Current signal name")
    voltage_v: str = Field(default="voltage_v", description="Voltage signal name")
    temperature_c: str = Field(default="temperature_c", description="Temperature signal name")
    state_on_off: str = Field(default="state_on_off", description="On/off state signal")
    trip_flag: str = Field(default="trip_flag", description="Trip flag signal")
    overload_flag: str = Field(default="overload_flag", description="Overload flag signal")
    protection_event: str = Field(default="protection_event", description="Protection event signal")
    reset_counter: str = Field(default="reset_counter", description="Reset counter signal")
    pwm_duty_pct: str = Field(default="pwm_duty_pct", description="PWM duty cycle signal")
    device_status: str = Field(default="device_status", description="Device status signal")


# Defaults for missing columns
_COLUMN_DEFAULTS: dict[str, Any] = {
    "state_on_off": True,
    "trip_flag": False,
    "overload_flag": False,
    "protection_event": ProtectionEvent.NONE.value,
    "reset_counter": 0,
    "pwm_duty_pct": 100.0,
    "device_status": DeviceStatus.OK.value,
}


class MeasurementReader:
    """Reads measurement files and produces VIP-compatible DataFrames.

    Supported formats:
      - .mf4 / .mdf  → ASAM MDF4 via asammdf
      - .csv          → pandas CSV reader
      - .parquet      → pandas Parquet reader

    The column mapping translates source signal names to VIP schema.
    For MDF4 files, channels can be selected by name or group/index.
    """

    def __init__(
        self,
        mapping: ColumnMapping | None = None,
        default_channel_id: str = "ch_01",
    ) -> None:
        self.mapping = mapping or ColumnMapping()
        self.default_channel_id = default_channel_id

    def read(self, path: str | Path) -> pd.DataFrame:
        """Read a measurement file and return a VIP telemetry DataFrame."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Measurement file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in (".mf4", ".mdf"):
            raw = self._read_mdf4(path)
        elif suffix == ".csv":
            raw = self._read_csv(path)
        elif suffix == ".parquet":
            raw = self._read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}  (expected .mf4, .mdf, .csv, .parquet)")

        df = self._apply_mapping(raw)
        log.info("Loaded %d rows from %s (%d columns mapped)", len(df), path.name, len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Format-specific readers
    # ------------------------------------------------------------------

    def _read_mdf4(self, path: Path) -> pd.DataFrame:
        """Read an MDF4 file using asammdf."""
        try:
            from asammdf import MDF
        except ImportError:
            raise ImportError(
                "asammdf is required for MDF4 files. Install with: pip install asammdf"
            )

        mdf = MDF(path)
        # Export all channels to a single DataFrame
        # asammdf's to_dataframe() merges all groups with interpolation
        df = mdf.to_dataframe(time_as_date=True)
        mdf.close()

        # asammdf puts time as the index — move to a column
        if df.index.name == "timestamps" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            # Rename the index column to match mapping
            if "timestamps" in df.columns:
                df = df.rename(columns={"timestamps": self.mapping.timestamp})

        log.info("MDF4: %d rows, %d channels from %s", len(df), len(df.columns), path.name)
        return df

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read a CSV file with timestamp parsing."""
        df = pd.read_csv(path)
        # Try to parse timestamp column
        ts_col = self.mapping.timestamp
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        return df

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        """Read a Parquet file."""
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Column mapping
    # ------------------------------------------------------------------

    def _apply_mapping(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map source columns to VIP schema and fill defaults for missing ones."""
        result = pd.DataFrame()
        mapping_dict = self.mapping.model_dump()

        for vip_col, src_col in mapping_dict.items():
            if src_col and src_col in raw.columns:
                result[vip_col] = raw[src_col].values
            elif vip_col == "channel_id":
                # If no channel_id column, use default
                result["channel_id"] = self.default_channel_id
            elif vip_col == "timestamp":
                # If timestamp is the index or missing, try to find one
                if isinstance(raw.index, pd.DatetimeIndex):
                    result["timestamp"] = raw.index.values
                else:
                    raise ValueError(
                        f"Timestamp column '{src_col}' not found in source data. "
                        f"Available columns: {list(raw.columns)}"
                    )
            elif vip_col in _COLUMN_DEFAULTS:
                result[vip_col] = _COLUMN_DEFAULTS[vip_col]
            else:
                log.warning("Column '%s' (mapped from '%s') not found — will be NaN", vip_col, src_col)
                result[vip_col] = np.nan

        # Ensure timestamp is datetime
        if "timestamp" in result.columns:
            if not pd.api.types.is_datetime64_any_dtype(result["timestamp"]):
                result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")

        # Coerce types
        for col in ("current_a", "voltage_v", "temperature_c", "pwm_duty_pct"):
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        for col in ("state_on_off", "trip_flag", "overload_flag"):
            if col in result.columns:
                result[col] = result[col].astype(bool)

        if "reset_counter" in result.columns:
            result["reset_counter"] = pd.to_numeric(result["reset_counter"], errors="coerce").fillna(0).astype(int)

        return result

    # ------------------------------------------------------------------
    # Multi-channel MDF4 helper
    # ------------------------------------------------------------------

    def read_multichannel(
        self,
        path: str | Path,
        channel_signals: dict[str, dict[str, str]],
    ) -> pd.DataFrame:
        """Read an MDF4/CSV/Parquet with multiple eFuse channels.

        Args:
            path: Path to the measurement file.
            channel_signals: Mapping from channel_id to per-signal source names.
                Example::

                    {
                        "ch_01": {"current_a": "IC1_Ch1_Current", "voltage_v": "IC1_Ch1_Voltage", ...},
                        "ch_02": {"current_a": "IC1_Ch2_Current", "voltage_v": "IC1_Ch2_Voltage", ...},
                    }

        Returns:
            A combined DataFrame with all channels in VIP schema (long format).
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".mf4", ".mdf"):
            raw = self._read_mdf4(path)
        elif suffix == ".csv":
            raw = self._read_csv(path)
        elif suffix == ".parquet":
            raw = self._read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        frames: list[pd.DataFrame] = []
        base_mapping = self.mapping.model_dump()

        for ch_id, sig_map in channel_signals.items():
            # Build a per-channel mapping
            ch_mapping = dict(base_mapping)
            ch_mapping.update(sig_map)
            mapping_obj = ColumnMapping(**ch_mapping)

            reader = MeasurementReader(mapping=mapping_obj, default_channel_id=ch_id)
            ch_df = reader._apply_mapping(raw)
            ch_df["channel_id"] = ch_id
            frames.append(ch_df)

        if not frames:
            raise ValueError("No channels specified in channel_signals")

        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values(["timestamp", "channel_id"], inplace=True)
        combined.reset_index(drop=True, inplace=True)

        log.info(
            "Multi-channel read: %d channels, %d total rows from %s",
            len(channel_signals), len(combined), path.name,
        )
        return combined
