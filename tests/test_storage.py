"""Tests for the storage writer — round-trip fidelity per format."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.config.models import StorageConfig
from src.storage.writer import StorageWriter


def _sample_df(n: int = 20) -> pd.DataFrame:
    t0 = datetime.now(tz=timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": [t0 + timedelta(milliseconds=i * 100) for i in range(n)],
            "channel_id": "ch_01",
            "current_a": np.random.default_rng(0).normal(5.0, 0.1, n),
            "voltage_v": 13.5,
            "list_col": [["a", "b"]] * n,
        }
    )


@pytest.fixture()
def parquet_writer(tmp_path):
    cfg = StorageConfig(output_dir=str(tmp_path), format="parquet")
    return StorageWriter(cfg)


@pytest.fixture()
def csv_writer(tmp_path):
    cfg = StorageConfig(output_dir=str(tmp_path), format="csv")
    return StorageWriter(cfg)


@pytest.fixture()
def json_writer(tmp_path):
    cfg = StorageConfig(output_dir=str(tmp_path), format="json")
    return StorageWriter(cfg)


def test_parquet_round_trip(parquet_writer):
    df = _sample_df()
    path = parquet_writer.write_telemetry(df, name="test")
    loaded = pd.read_parquet(path)
    assert len(loaded) == len(df)
    assert set(loaded.columns) == set(df.columns)


def test_csv_round_trip(csv_writer):
    df = _sample_df()
    path = csv_writer.write_telemetry(df, name="test")
    loaded = pd.read_csv(path)
    assert len(loaded) == len(df)


def test_json_round_trip(json_writer):
    df = _sample_df()
    path = json_writer.write_telemetry(df, name="test")
    loaded = pd.read_json(path)
    assert len(loaded) == len(df)


def test_list_col_survives_parquet(parquet_writer):
    """List columns serialized with json.dumps should survive round-trip."""
    df = _sample_df()
    path = parquet_writer.write_telemetry(df, name="list_test")
    loaded = pd.read_parquet(path)
    # Should be a JSON string representing the list
    parsed = json.loads(loaded["list_col"].iloc[0])
    assert parsed == ["a", "b"]


def test_write_alerts_creates_json(tmp_path):
    cfg = StorageConfig(output_dir=str(tmp_path))
    writer = StorageWriter(cfg)
    alerts = [{"ts": "2024-01-01", "fault": "overload_spike", "score": 0.9}]
    path = writer.write_alerts(alerts)
    with open(path) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["fault"] == "overload_spike"


def test_write_alerts_append(tmp_path):
    """Subsequent write_alerts should append, not overwrite."""
    cfg = StorageConfig(output_dir=str(tmp_path))
    writer = StorageWriter(cfg)
    writer.write_alerts([{"id": 1}])
    writer.write_alerts([{"id": 2}])
    path = tmp_path / "alerts.json"
    with open(path) as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["id"] == 1
    assert data[1]["id"] == 2


def test_write_features(parquet_writer):
    df = _sample_df()
    path = parquet_writer.write_features(df, name="feat")
    assert path.exists()


def test_write_scored(parquet_writer):
    df = _sample_df()
    path = parquet_writer.write_scored(df, name="scored")
    assert path.exists()


def test_output_dir_created_if_missing(tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    cfg = StorageConfig(output_dir=str(nested))
    writer = StorageWriter(cfg)
    df = _sample_df(5)
    path = writer.write_telemetry(df)
    assert path.exists()
