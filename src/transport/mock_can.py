"""Transport abstraction — mock CAN bus / replay / streaming.

Provides a common interface for telemetry delivery so the downstream
pipeline doesn't care whether data comes from simulation, replay, or
a real CAN interface.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

import pandas as pd

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
