"""Transport abstraction — CAN / XCP / replay / streaming.

Provides a common interface for telemetry delivery so the downstream
pipeline doesn't care whether data comes from simulation, replay, a
real CAN interface, or an XCP measurement session.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

import pandas as pd

from src.schemas.telemetry import SourceProtocol
from src.utils.logging import get_logger

log = get_logger(__name__)


class TransportBase(ABC):
    """Abstract base for telemetry sources."""

    @abstractmethod
    def stream(self) -> Iterator[dict]:
        """Yield one telemetry row at a time."""
        ...

    @abstractmethod
    def batch(self, size: int) -> list[dict]:
        """Return up to `size` rows in one call."""
        ...


class DataFrameTransport(TransportBase):
    """Wraps a pre-generated DataFrame as a streaming source."""

    def __init__(self, df: pd.DataFrame, realtime: bool = False, speed: float = 1.0) -> None:
        self._df = df
        self._realtime = realtime
        self._speed = max(speed, 0.01)
        self._idx = 0

    def stream(self) -> Iterator[dict]:
        prev_ts = None
        for _, row in self._df.iterrows():
            record = row.to_dict()
            if self._realtime and prev_ts is not None:
                dt = (record["timestamp"] - prev_ts).total_seconds() / self._speed
                if dt > 0:
                    time.sleep(dt)
            prev_ts = record["timestamp"]
            yield record

    def batch(self, size: int) -> list[dict]:
        end = min(self._idx + size, len(self._df))
        if self._idx >= end:
            return []
        chunk = self._df.iloc[self._idx:end].to_dict(orient="records")
        self._idx = end
        return chunk

    def reset(self) -> None:
        self._idx = 0

    @property
    def exhausted(self) -> bool:
        return self._idx >= len(self._df)


class ReplayTransport(DataFrameTransport):
    """Replays telemetry from a Parquet / CSV file."""

    def __init__(self, path: str, realtime: bool = False, speed: float = 1.0) -> None:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, parse_dates=["timestamp"])
        log.info("Loaded %d rows from %s", len(df), path)
        super().__init__(df, realtime=realtime, speed=speed)


class XcpTransport(DataFrameTransport):
    """Simulates an XCP DAQ session with dual-raster measurement.

    In real hardware, XCP provides direct memory reads at configurable
    DAQ rates — typically 10ms for fast signals (current, voltage) and
    50ms for slow signals (temperature, status).  This transport tags
    each row with the source protocol for downstream handling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fast_raster_ms: float = 10.0,
        slow_raster_ms: float = 50.0,
        realtime: bool = False,
        speed: float = 1.0,
    ) -> None:
        # Tag rows with source protocol
        df = df.copy()
        df["source_protocol"] = SourceProtocol.XCP.value
        super().__init__(df, realtime=realtime, speed=speed)
        self.fast_raster_ms = fast_raster_ms
        self.slow_raster_ms = slow_raster_ms
        log.info(
            "XCP transport: fast=%gms, slow=%gms, %d rows",
            fast_raster_ms, slow_raster_ms, len(df),
        )


class CanTransport(DataFrameTransport):
    """Tags rows with CAN source protocol for production vehicle data."""

    def __init__(self, df: pd.DataFrame, realtime: bool = False, speed: float = 1.0) -> None:
        df = df.copy()
        df["source_protocol"] = SourceProtocol.CAN.value
        super().__init__(df, realtime=realtime, speed=speed)
