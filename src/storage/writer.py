"""Storage — persist telemetry, features, alerts to local files.

Supports Parquet (default), CSV, and JSON.
Optional backend sync is stubbed for future implementation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.config.models import StorageConfig
from src.utils.logging import get_logger

log = get_logger(__name__)


class StorageWriter:
    """Writes DataFrames and alert dicts to local storage."""

    def __init__(self, config: StorageConfig | None = None, disk_min_free_mb: int = 0) -> None:
        self.cfg = config or StorageConfig()
        self._out = Path(self.cfg.output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._disk_min_free_mb = disk_min_free_mb

    def write_telemetry(self, df: pd.DataFrame, name: str = "telemetry") -> Path:
        return self._write_df(df, name)

    def write_features(self, df: pd.DataFrame, name: str = "features") -> Path:
        return self._write_df(df, name)

    def write_labels(self, df: pd.DataFrame, name: str = "labels") -> Path:
        return self._write_df(df, name)

    def write_scored(self, df: pd.DataFrame, name: str = "scored") -> Path:
        return self._write_df(df, name)

    def check_disk_space(self) -> bool:
        """Return True if disk has enough free space (or threshold is 0)."""
        if self._disk_min_free_mb <= 0:
            return True
        try:
            usage = shutil.disk_usage(self._out)
            free_mb = usage.free / (1024 * 1024)
            if free_mb < self._disk_min_free_mb:
                log.warning(
                    "Low disk space: %.0f MB free (threshold: %d MB) — skipping write",
                    free_mb, self._disk_min_free_mb,
                )
                return False
        except OSError:
            pass  # can't check → allow write
        return True

    def write_alerts(self, alerts: list[dict], name: str = "alerts") -> Path | None:
        p = self._out / f"{name}.json"
        if not self.check_disk_space():
            return None
        # Append-safe: read existing, extend, rewrite
        existing = []
        if p.exists():
            with open(p) as f:
                existing = json.load(f)
        existing.extend(alerts)
        with open(p, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        log.info("Wrote %d alerts to %s", len(alerts), p)
        return p

    def _write_df(self, df: pd.DataFrame, name: str) -> Path | None:
        if not self.check_disk_space():
            return None
        # Serialize list columns as JSON strings for round-trip fidelity
        df_out = df.copy()
        for col in df_out.columns:
            if df_out[col].apply(lambda x: isinstance(x, list)).any():
                df_out[col] = df_out[col].apply(json.dumps)

        fmt = self.cfg.format
        if fmt == "parquet":
            p = self._out / f"{name}.parquet"
            df_out.to_parquet(p, index=False)
        elif fmt == "csv":
            p = self._out / f"{name}.csv"
            df_out.to_csv(p, index=False)
        else:
            p = self._out / f"{name}.json"
            df_out.to_json(p, orient="records", date_format="iso", indent=2)
        log.info("Wrote %d rows to %s", len(df_out), p)
        return p
