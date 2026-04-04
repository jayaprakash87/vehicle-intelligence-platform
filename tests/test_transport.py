"""Tests for the transport layer."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.transport.mock_can import DataFrameTransport, ReplayTransport


def _make_df(n: int = 100) -> pd.DataFrame:
    t0 = datetime.now(tz=timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": [t0 + timedelta(milliseconds=i * 100) for i in range(n)],
            "channel_id": "ch_01",
            "current_a": np.random.default_rng(42).normal(5.0, 0.2, n),
            "voltage_v": 13.5,
        }
    )


def test_batch_returns_correct_size():
    df = _make_df(100)
    t = DataFrameTransport(df)
    batch = t.batch(30)
    assert len(batch) == 30
    # Each item is a dict
    assert isinstance(batch[0], dict)


def test_batch_exhausts_transport():
    df = _make_df(50)
    t = DataFrameTransport(df)
    b1 = t.batch(30)
    assert len(b1) == 30
    assert not t.exhausted
    b2 = t.batch(30)
    assert len(b2) == 20  # only 20 remain
    assert t.exhausted
    b3 = t.batch(30)
    assert len(b3) == 0


def test_batch_empty_transport():
    df = _make_df(0)
    t = DataFrameTransport(df)
    assert t.exhausted
    assert t.batch(10) == []


def test_reset_restores_position():
    df = _make_df(50)
    t = DataFrameTransport(df)
    t.batch(50)
    assert t.exhausted
    t.reset()
    assert not t.exhausted
    b = t.batch(50)
    assert len(b) == 50


def test_stream_yields_all_rows():
    df = _make_df(20)
    t = DataFrameTransport(df)
    rows = list(t.stream())
    assert len(rows) == 20
    assert all(isinstance(r, dict) for r in rows)
    assert rows[0]["channel_id"] == "ch_01"


def test_exhausted_property():
    df = _make_df(10)
    t = DataFrameTransport(df)
    assert not t.exhausted
    t.batch(5)
    assert not t.exhausted
    t.batch(5)
    assert t.exhausted


def test_replay_transport_parquet(tmp_path):
    df = _make_df(30)
    p = tmp_path / "test.parquet"
    df.to_parquet(p, index=False)
    t = ReplayTransport(str(p))
    batch = t.batch(30)
    assert len(batch) == 30


def test_replay_transport_csv(tmp_path):
    df = _make_df(20)
    p = tmp_path / "test.csv"
    df.to_csv(p, index=False)
    t = ReplayTransport(str(p))
    batch = t.batch(20)
    assert len(batch) == 20
