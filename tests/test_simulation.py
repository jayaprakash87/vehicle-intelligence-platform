"""Tests for the simulation generator."""

import pandas as pd

from src.config.models import SimulationConfig
from src.schemas.telemetry import ChannelMeta, FaultInjection, FaultType, ProtectionEvent
from src.simulation.generator import TelemetryGenerator


def _make_config(**overrides) -> SimulationConfig:
    defaults = dict(
        scenario_id="test",
        name="Test",
        duration_s=10.0,
        sample_interval_ms=100.0,
        seed=42,
        channels=[ChannelMeta(channel_id="ch_01", load_name="test_load", nominal_current_a=5.0)],
        fault_injections=[],
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def test_generate_nominal():
    cfg = _make_config()
    gen = TelemetryGenerator(cfg)
    telem_df, labels_df = gen.generate()
    assert len(telem_df) == 100  # 10s / 0.1s
    assert "current_a" in telem_df.columns
    assert len(labels_df) == 0  # no faults


def test_generate_with_fault():
    cfg = _make_config(
        fault_injections=[
            FaultInjection(
                channel_id="ch_01",
                fault_type=FaultType.OVERLOAD_SPIKE,
                start_s=2.0,
                duration_s=2.0,
                intensity=0.8,
            )
        ]
    )
    gen = TelemetryGenerator(cfg)
    telem_df, labels_df = gen.generate()
    assert len(labels_df) > 0
    assert labels_df["fault_type"].iloc[0] == FaultType.OVERLOAD_SPIKE.value


def test_reproducible_with_seed():
    cfg = _make_config(seed=99)
    gen1 = TelemetryGenerator(cfg)
    gen2 = TelemetryGenerator(cfg)
    df1, _ = gen1.generate()
    df2, _ = gen2.generate()
    # Timestamps differ because they use datetime.now(); compare signal columns only
    cols = [c for c in df1.columns if c != "timestamp"]
    pd.testing.assert_frame_equal(df1[cols], df2[cols])


def test_multiple_channels():
    cfg = _make_config(
        channels=[
            ChannelMeta(channel_id="ch_01", load_name="a", nominal_current_a=5.0),
            ChannelMeta(channel_id="ch_02", load_name="b", nominal_current_a=10.0),
        ]
    )
    gen = TelemetryGenerator(cfg)
    telem_df, _ = gen.generate()
    assert set(telem_df["channel_id"].unique()) == {"ch_01", "ch_02"}
    assert len(telem_df) == 200  # 100 per channel


# ---------------------------------------------------------------------------
# Dual ADC quantization
# ---------------------------------------------------------------------------

def test_voltage_adc_quantization():
    """Voltage signal should be quantized to voltage_adc_bits resolution."""
    ch = ChannelMeta(
        channel_id="ch_01",
        load_name="test",
        nominal_current_a=5.0,
        voltage_adc_bits=10,
    )
    cfg = _make_config(channels=[ch], duration_s=5.0, sample_interval_ms=50.0)
    gen = TelemetryGenerator(cfg)
    df, _ = gen.generate()
    voltages = df["voltage_v"].dropna().values
    # With 10-bit ADC and ~40.5V range, LSB ≈ 0.0396V
    v_lsb = (ch.nominal_voltage_v * 3.0) / (2 ** ch.voltage_adc_bits)
    # Verify all voltages are near multiples of LSB
    residuals = voltages / v_lsb - (voltages / v_lsb).round()
    assert abs(residuals).max() < 0.01, "Voltage not quantized to voltage_adc_bits"


def test_current_adc_quantization():
    """Current signal should be quantized per current_adc_bits."""
    ch = ChannelMeta(
        channel_id="ch_01",
        load_name="test",
        nominal_current_a=5.0,
        max_current_a=20.0,
        current_adc_bits=10,  # coarse — easy to verify
    )
    cfg = _make_config(channels=[ch], duration_s=5.0, sample_interval_ms=50.0)
    gen = TelemetryGenerator(cfg)
    df, _ = gen.generate()
    currents = df["current_a"].dropna().values
    adc_range = ch.max_current_a * 1.5
    lsb = adc_range / (2 ** ch.current_adc_bits)
    residuals = currents / lsb - (currents / lsb).round()
    assert abs(residuals).max() < 0.01, "Current not quantized to current_adc_bits"


# ---------------------------------------------------------------------------
# F(i,t) protection integration
# ---------------------------------------------------------------------------

def test_fit_protection_trips_on_overload():
    """Overload spike fault should still cause a trip via F(i,t) model."""
    ch = ChannelMeta(
        channel_id="ch_01",
        load_name="test",
        nominal_current_a=5.0,
        max_current_a=20.0,
        fuse_rating_a=15.0,
        cooldown_s=0.5,
        max_retries=2,
    )
    cfg = _make_config(
        channels=[ch],
        duration_s=10.0,
        sample_interval_ms=50.0,
        fault_injections=[
            FaultInjection(
                channel_id="ch_01",
                fault_type=FaultType.OVERLOAD_SPIKE,
                start_s=2.0,
                duration_s=4.0,
                intensity=0.9,
            )
        ],
    )
    gen = TelemetryGenerator(cfg)
    df, labels = gen.generate()
    # Trip flag should be set at some point during the fault
    assert df["trip_flag"].any(), "F(i,t) protection should trip on overload"
    # After max retries, channel should latch off (current near zero)
    tripped_rows = df[df["trip_flag"]]
    assert (tripped_rows["current_a"].abs() < 1.0).any(), "Latch-off should have near-zero current"


def test_protection_event_tagged_on_overload():
    """protection_event column should carry SCP/I2T/LATCH_OFF, not just 'none'."""
    ch = ChannelMeta(
        channel_id="ch_01",
        load_name="test",
        nominal_current_a=5.0,
        max_current_a=20.0,
        fuse_rating_a=15.0,
        cooldown_s=0.5,
        max_retries=2,
    )
    cfg = _make_config(
        channels=[ch],
        duration_s=10.0,
        sample_interval_ms=50.0,
        fault_injections=[
            FaultInjection(
                channel_id="ch_01",
                fault_type=FaultType.OVERLOAD_SPIKE,
                start_s=2.0,
                duration_s=4.0,
                intensity=0.9,
            )
        ],
    )
    gen = TelemetryGenerator(cfg)
    df, labels = gen.generate()

    assert "protection_event" in df.columns, "DataFrame must include protection_event"
    events = set(df["protection_event"].unique())
    # Should have at least 'none' and one of the trip types
    assert ProtectionEvent.NONE.value in events
    non_none = events - {ProtectionEvent.NONE.value}
    assert len(non_none) > 0, "Overload should produce at least one protection event"
    # All non-none events should be valid ProtectionEvent values
    valid_values = {e.value for e in ProtectionEvent}
    assert non_none <= valid_values, f"Unexpected protection events: {non_none - valid_values}"


def test_nominal_has_no_protection_events():
    """Nominal scenario should have protection_event = 'none' everywhere."""
    cfg = _make_config()
    gen = TelemetryGenerator(cfg)
    df, _ = gen.generate()
    assert (df["protection_event"] == ProtectionEvent.NONE.value).all()


def test_catalog_propagates_dual_adc():
    """build_channels should propagate current/voltage ADC bits from catalog."""
    from src.config.catalog import build_channels, get_profile, sedan_topology, EFUSE_CATALOG
    from src.schemas.telemetry import EFuseFamily, ZoneController

    zones, specs = sedan_topology()
    channels = build_channels(zones, specs)
    # Pick a channel and verify it inherited the catalog's ADC settings
    ch = channels[0]
    profile = EFUSE_CATALOG[ch.efuse_family]
    assert ch.current_adc_bits == profile.current_adc_bits
    assert ch.voltage_adc_bits == profile.voltage_adc_bits
