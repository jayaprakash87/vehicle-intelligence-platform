"""Tests for the measurement file reader and column mapping."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ingestion.reader import ColumnMapping, MeasurementReader
from src.schemas.telemetry import DeviceStatus, ProtectionEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv_data(n: int = 50) -> pd.DataFrame:
    """Create a minimal DataFrame that would come from a CSV."""
    rng = np.random.default_rng(42)
    t0 = datetime.now(tz=timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": [t0 + timedelta(milliseconds=i * 100) for i in range(n)],
            "channel_id": "ch_01",
            "current_a": rng.normal(5.0, 0.2, n),
            "voltage_v": rng.normal(13.5, 0.05, n),
            "temperature_c": 25.0 + rng.normal(0, 0.5, n),
        }
    )


def _make_oem_data(n: int = 50) -> pd.DataFrame:
    """Create data with OEM-specific signal names."""
    rng = np.random.default_rng(42)
    t0 = datetime.now(tz=timezone.utc)
    return pd.DataFrame(
        {
            "Time_UTC": [t0 + timedelta(milliseconds=i * 100) for i in range(n)],
            "IC1_Ch1_Isense": rng.normal(5.0, 0.2, n),
            "IC1_Ch1_Vsense": rng.normal(13.5, 0.05, n),
            "IC1_Ch1_Tjunction": 25.0 + rng.normal(0, 0.5, n),
            "IC1_Ch2_Isense": rng.normal(10.0, 0.3, n),
            "IC1_Ch2_Vsense": rng.normal(13.5, 0.05, n),
            "IC1_Ch2_Tjunction": 30.0 + rng.normal(0, 0.5, n),
        }
    )


# ---------------------------------------------------------------------------
# ColumnMapping
# ---------------------------------------------------------------------------


class TestColumnMapping:
    def test_defaults(self):
        m = ColumnMapping()
        assert m.timestamp == "timestamp"
        assert m.current_a == "current_a"
        assert m.voltage_v == "voltage_v"

    def test_custom_mapping(self):
        m = ColumnMapping(current_a="IC1_Isense", voltage_v="IC1_Vsense")
        assert m.current_a == "IC1_Isense"
        assert m.voltage_v == "IC1_Vsense"
        # Others stay default
        assert m.timestamp == "timestamp"

    def test_model_dump(self):
        m = ColumnMapping()
        d = m.model_dump()
        assert "current_a" in d
        assert "timestamp" in d
        assert len(d) == 12  # all telemetry columns


# ---------------------------------------------------------------------------
# MeasurementReader — CSV
# ---------------------------------------------------------------------------


class TestReaderCsv:
    def test_read_csv_with_defaults(self, tmp_path):
        """Standard VIP column names → direct passthrough."""
        df = _make_csv_data()
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        reader = MeasurementReader()
        result = reader.read(str(csv_path))

        assert len(result) == 50
        assert "current_a" in result.columns
        assert "voltage_v" in result.columns
        assert "timestamp" in result.columns
        assert "channel_id" in result.columns

    def test_read_csv_with_mapping(self, tmp_path):
        """OEM signal names mapped to VIP columns."""
        df = _make_oem_data()
        csv_path = tmp_path / "oem_data.csv"
        df.to_csv(csv_path, index=False)

        mapping = ColumnMapping(
            timestamp="Time_UTC",
            current_a="IC1_Ch1_Isense",
            voltage_v="IC1_Ch1_Vsense",
            temperature_c="IC1_Ch1_Tjunction",
        )
        reader = MeasurementReader(mapping=mapping, default_channel_id="ch_headlamp")
        result = reader.read(str(csv_path))

        assert len(result) == 50
        assert result["channel_id"].iloc[0] == "ch_headlamp"
        assert abs(result["current_a"].mean() - 5.0) < 1.0
        assert abs(result["voltage_v"].mean() - 13.5) < 1.0

    def test_missing_columns_get_defaults(self, tmp_path):
        """Columns absent from source get filled with defaults."""
        df = _make_csv_data()
        csv_path = tmp_path / "partial.csv"
        df.to_csv(csv_path, index=False)

        reader = MeasurementReader()
        result = reader.read(str(csv_path))

        # trip_flag, protection_event, etc. should have defaults
        assert (~result["trip_flag"]).all()
        assert (result["protection_event"] == ProtectionEvent.NONE.value).all()
        assert (result["reset_counter"] == 0).all()
        assert (result["pwm_duty_pct"] == 100.0).all()
        assert (result["device_status"] == DeviceStatus.OK.value).all()

    def test_file_not_found_raises(self):
        reader = MeasurementReader()
        with pytest.raises(FileNotFoundError):
            reader.read("/nonexistent/file.csv")

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "data.xlsx"
        p.write_text("dummy")
        reader = MeasurementReader()
        with pytest.raises(ValueError, match="Unsupported"):
            reader.read(str(p))


# ---------------------------------------------------------------------------
# MeasurementReader — Parquet
# ---------------------------------------------------------------------------


class TestReaderParquet:
    def test_read_parquet(self, tmp_path):
        df = _make_csv_data()
        pq_path = tmp_path / "test.parquet"
        df.to_parquet(pq_path, index=False)

        reader = MeasurementReader()
        result = reader.read(str(pq_path))

        assert len(result) == 50
        assert "current_a" in result.columns

    def test_read_parquet_with_mapping(self, tmp_path):
        df = _make_oem_data()
        pq_path = tmp_path / "oem.parquet"
        df.to_parquet(pq_path, index=False)

        mapping = ColumnMapping(
            timestamp="Time_UTC",
            current_a="IC1_Ch1_Isense",
            voltage_v="IC1_Ch1_Vsense",
            temperature_c="IC1_Ch1_Tjunction",
        )
        reader = MeasurementReader(mapping=mapping, default_channel_id="ch_01")
        result = reader.read(str(pq_path))

        assert len(result) == 50
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


# ---------------------------------------------------------------------------
# Multi-channel reading
# ---------------------------------------------------------------------------


class TestMultiChannel:
    def test_read_multichannel(self, tmp_path):
        """Read one file with signals for 2 channels."""
        df = _make_oem_data()
        csv_path = tmp_path / "multi.csv"
        df.to_csv(csv_path, index=False)

        reader = MeasurementReader(
            mapping=ColumnMapping(timestamp="Time_UTC"),
        )
        channel_signals = {
            "ch_01": {
                "current_a": "IC1_Ch1_Isense",
                "voltage_v": "IC1_Ch1_Vsense",
                "temperature_c": "IC1_Ch1_Tjunction",
            },
            "ch_02": {
                "current_a": "IC1_Ch2_Isense",
                "voltage_v": "IC1_Ch2_Vsense",
                "temperature_c": "IC1_Ch2_Tjunction",
            },
        }
        result = reader.read_multichannel(str(csv_path), channel_signals)

        assert result["channel_id"].nunique() == 2
        assert len(result) == 100  # 50 per channel
        assert set(result["channel_id"].unique()) == {"ch_01", "ch_02"}
        # ch_02 should have higher current (~10A vs ~5A)
        ch2 = result[result["channel_id"] == "ch_02"]
        assert ch2["current_a"].mean() > 8.0


# ---------------------------------------------------------------------------
# MDF4 (mocked)
# ---------------------------------------------------------------------------


class TestReaderMdf4:
    def test_read_mdf4_mocked(self, tmp_path):
        """MDF4 reading via asammdf — mock the MDF class."""
        rng = np.random.default_rng(42)
        n = 30
        t0 = datetime.now(tz=timezone.utc)
        mock_df = pd.DataFrame(
            {
                "IC1_Current": rng.normal(5.0, 0.2, n),
                "IC1_Voltage": rng.normal(13.5, 0.05, n),
                "IC1_Temperature": 25.0 + rng.normal(0, 0.5, n),
            },
            index=pd.DatetimeIndex(
                [t0 + timedelta(milliseconds=i * 100) for i in range(n)],
                name="timestamps",
            ),
        )

        mock_mdf = MagicMock()
        mock_mdf.to_dataframe.return_value = mock_df

        # Write a dummy .mf4 file so Path.exists() passes
        mf4_path = tmp_path / "recording.mf4"
        mf4_path.write_bytes(b"dummy")

        mapping = ColumnMapping(
            current_a="IC1_Current",
            voltage_v="IC1_Voltage",
            temperature_c="IC1_Temperature",
        )
        reader = MeasurementReader(mapping=mapping, default_channel_id="ch_01")

        with patch("src.ingestion.reader.MDF", return_value=mock_mdf, create=True):
            # Patch the import inside _read_mdf4
            import src.ingestion.reader as reader_mod

            getattr(reader_mod, "_mdf_import_done", False)
            with patch.object(reader_mod, "_read_mdf4_import", create=True):
                # Directly patch asammdf.MDF in the function's local scope
                with patch.dict(
                    "sys.modules", {"asammdf": MagicMock(MDF=MagicMock(return_value=mock_mdf))}
                ):
                    result = reader.read(str(mf4_path))

        assert len(result) == 30
        assert "current_a" in result.columns
        assert abs(result["current_a"].mean() - 5.0) < 1.0
        mock_mdf.close.assert_called_once()


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


class TestTypeCoercion:
    def test_numeric_coercion(self, tmp_path):
        """String values in numeric columns should be coerced."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=5, freq="100ms", tz="UTC"),
                "channel_id": "ch_01",
                "current_a": ["5.1", "5.2", "bad", "5.4", "5.5"],
                "voltage_v": ["13.5", "13.4", "13.6", "13.5", "13.4"],
                "temperature_c": ["25.0", "25.1", "25.2", "25.3", "25.4"],
            }
        )
        csv_path = tmp_path / "string_nums.csv"
        df.to_csv(csv_path, index=False)

        reader = MeasurementReader()
        result = reader.read(str(csv_path))

        assert pd.api.types.is_float_dtype(result["current_a"])
        assert pd.api.types.is_float_dtype(result["voltage_v"])
        # "bad" should become NaN
        assert result["current_a"].isna().sum() == 1


class TestEndToEnd:
    def test_csv_through_normalizer(self, tmp_path):
        """Reader output should be compatible with the Normalizer."""
        from src.ingestion.normalizer import Normalizer
        from src.config.models import NormalizerConfig

        df = _make_csv_data()
        csv_path = tmp_path / "e2e.csv"
        df.to_csv(csv_path, index=False)

        reader = MeasurementReader()
        raw = reader.read(str(csv_path))

        norm = Normalizer(NormalizerConfig())
        result = norm.normalize(raw)

        assert len(result) == 50
        assert "missing_rate" in result.columns
