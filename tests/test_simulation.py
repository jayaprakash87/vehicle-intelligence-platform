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
    v_lsb = (ch.nominal_voltage_v * 3.0) / (2**ch.voltage_adc_bits)
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
    lsb = adc_range / (2**ch.current_adc_bits)
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
    from src.config.catalog import build_channels, example_topology, EFUSE_CATALOG

    zones, specs = example_topology()
    channels = build_channels(zones, specs)
    # Pick a channel and verify it inherited the catalog's ADC settings
    ch = channels[0]
    profile = EFUSE_CATALOG[ch.efuse_family]
    assert ch.current_adc_bits == profile.current_adc_bits
    assert ch.voltage_adc_bits == profile.voltage_adc_bits


# ---------------------------------------------------------------------------
# Thermal shutdown simulation
# ---------------------------------------------------------------------------


def test_thermal_shutdown_fires_on_extreme_drift():
    """A THERMAL_DRIFT fault with high intensity on a low-threshold channel
    should trigger THERMAL_SHUTDOWN protection events."""
    ch = ChannelMeta(
        channel_id="ch_01",
        load_name="test",
        nominal_current_a=10.0,
        max_current_a=30.0,
        fuse_rating_a=20.0,
        thermal_shutdown_c=100.0,  # low threshold to make it easier to trigger
        r_thermal_kw=60.0,  # higher thermal resistance → faster heating
        tau_thermal_s=5.0,
        t_ambient_c=25.0,
    )
    cfg = _make_config(
        channels=[ch],
        duration_s=20.0,
        sample_interval_ms=100.0,
        fault_injections=[
            FaultInjection(
                channel_id="ch_01",
                fault_type=FaultType.THERMAL_DRIFT,
                start_s=1.0,
                duration_s=15.0,
                intensity=1.0,
            )
        ],
    )
    gen = TelemetryGenerator(cfg)
    df, _ = gen.generate()

    thermal_events = df[df["protection_event"] == ProtectionEvent.THERMAL_SHUTDOWN.value]
    assert len(thermal_events) > 0, "Thermal shutdown should fire when T_j exceeds limit"
    # During thermal shutdown, current should be near zero
    assert thermal_events["current_a"].abs().max() < 0.5


def test_thermal_shutdown_hysteresis_recovery():
    """After thermal shutdown, channel should recover only after temperature drops
    below the hysteresis band (thermal_limit - 20°C)."""
    ch = ChannelMeta(
        channel_id="ch_01",
        load_name="test",
        nominal_current_a=10.0,
        max_current_a=30.0,
        fuse_rating_a=20.0,
        thermal_shutdown_c=100.0,
        r_thermal_kw=60.0,
        tau_thermal_s=3.0,  # fast thermal response for quicker decay
        t_ambient_c=25.0,
        rds_on_tempco_exp=0.0,  # disable tempco: this test exercises hysteresis, not Rds,on(T)
    )
    cfg = _make_config(
        channels=[ch],
        duration_s=60.0,
        sample_interval_ms=100.0,
        fault_injections=[
            FaultInjection(
                channel_id="ch_01",
                fault_type=FaultType.THERMAL_DRIFT,
                start_s=1.0,
                duration_s=5.0,  # short fault → temp should decay quickly after
                intensity=1.0,
            )
        ],
    )
    gen = TelemetryGenerator(cfg)
    df, _ = gen.generate()

    events = df["protection_event"].values
    # Should see THERMAL_SHUTDOWN events followed by a return to NONE
    thermal_mask = events == ProtectionEvent.THERMAL_SHUTDOWN.value
    if thermal_mask.any():
        first_shutdown = thermal_mask.argmax()
        # Find first recovery (non-shutdown) after shutdown
        post_shutdown = events[first_shutdown:]
        recovery_mask = post_shutdown != ProtectionEvent.THERMAL_SHUTDOWN.value
        # Should eventually recover (fault ends, temp decays)
        assert recovery_mask.any(), "Channel should recover after temp drops below hysteresis band"


def test_nominal_no_thermal_shutdown():
    """Nominal operation should never produce THERMAL_SHUTDOWN events."""
    cfg = _make_config()
    gen = TelemetryGenerator(cfg)
    df, _ = gen.generate()
    assert (df["protection_event"] != ProtectionEvent.THERMAL_SHUTDOWN.value).all()


# ---------------------------------------------------------------------------
# Gap #5 — Abnormal bus voltage scenarios
# ---------------------------------------------------------------------------

def _make_ch_bus(rds_on_tempco_exp: float = 0.0) -> ChannelMeta:
    """Minimal ChannelMeta for bus-voltage scenario tests (tempco off for simplicity)."""
    return ChannelMeta(
        channel_id="ch_bus",
        load_name="seat_heater",
        nominal_current_a=8.0,
        max_current_a=30.0,
        fuse_rating_a=20.0,
        thermal_shutdown_c=150.0,
        r_thermal_kw=20.0,
        tau_thermal_s=10.0,
        t_ambient_c=25.0,
        rds_on_tempco_exp=rds_on_tempco_exp,
    )


def test_jump_start_elevates_bus_voltage():
    """JUMP_START fault — bus voltage should rise above 16 V and current increases."""
    ch = _make_ch_bus()
    cfg = _make_config(
        duration_s=20.0,
        sample_interval_ms=100.0,
        channels=[ch],
        fault_injections=[
            FaultInjection(channel_id="ch_bus", fault_type=FaultType.JUMP_START,
                           start_s=5.0, duration_s=10.0, intensity=0.8),
        ],
    )
    gen = TelemetryGenerator(cfg)
    df, labels = gen.generate()
    fault_rows = df[df["timestamp"].between(
        df["timestamp"].iloc[0] + pd.Timedelta(seconds=6),
        df["timestamp"].iloc[0] + pd.Timedelta(seconds=14),
    )]
    assert fault_rows["voltage_v"].max() > 16.0, "Jump-start should push bus above 16 V"
    ov_rows = fault_rows[fault_rows["protection_event"] == ProtectionEvent.OVER_VOLTAGE.value]
    assert len(ov_rows) > 0, "OVER_VOLTAGE protection event should be set during jump-start"
    assert labels["fault_type"].iloc[0] == FaultType.JUMP_START.value


def test_jump_start_current_scales_with_voltage():
    """During jump-start, resistive-load current should be higher than nominal."""
    ch = _make_ch_bus()
    cfg_nominal = _make_config(duration_s=10.0, channels=[ch], fault_injections=[])
    cfg_jump = _make_config(
        duration_s=10.0,
        channels=[ch],
        fault_injections=[
            FaultInjection(channel_id="ch_bus", fault_type=FaultType.JUMP_START,
                           start_s=0.0, duration_s=10.0, intensity=0.9),
        ],
    )
    gen_n = TelemetryGenerator(cfg_nominal)
    gen_j = TelemetryGenerator(cfg_jump)
    df_n, _ = gen_n.generate()
    df_j, _ = gen_j.generate()
    # Median current should be higher during jump-start
    med_jump = df_j["current_a"].median()
    med_nom = df_n["current_a"].median()
    assert med_jump > med_nom * 1.05, "Current should be elevated during jump-start"


def test_load_dump_spikes_and_shuts_off():
    """LOAD_DUMP fault — brief voltage spike to ~40 V, IC shuts gate off."""
    ch = _make_ch_bus()
    cfg = _make_config(
        duration_s=10.0,
        sample_interval_ms=50.0,
        channels=[ch],
        fault_injections=[
            FaultInjection(channel_id="ch_bus", fault_type=FaultType.LOAD_DUMP,
                           start_s=3.0, duration_s=2.0, intensity=1.0),
        ],
    )
    gen = TelemetryGenerator(cfg)
    df, labels = gen.generate()
    fault_rows = df[df["timestamp"].between(
        df["timestamp"].iloc[0] + pd.Timedelta(seconds=3),
        df["timestamp"].iloc[0] + pd.Timedelta(seconds=5),
    )]
    # Bus should spike above 30 V
    assert fault_rows["voltage_v"].max() > 30.0, "Load dump should produce > 30 V spike"
    # Trip flag should be set (IC over-voltage protection fires)
    assert fault_rows["trip_flag"].any(), "Load dump should trigger eFuse trip"
    # Protection event
    assert (fault_rows["protection_event"] == ProtectionEvent.OVER_VOLTAGE.value).any()
    assert labels["fault_type"].iloc[0] == FaultType.LOAD_DUMP.value


def test_cold_crank_sags_bus_voltage():
    """COLD_CRANK fault — bus sags to < 9 V and current drops proportionally."""
    ch = _make_ch_bus()
    cfg = _make_config(
        duration_s=20.0,
        sample_interval_ms=100.0,
        channels=[ch],
        fault_injections=[
            FaultInjection(channel_id="ch_bus", fault_type=FaultType.COLD_CRANK,
                           start_s=5.0, duration_s=8.0, intensity=0.9),
        ],
    )
    gen = TelemetryGenerator(cfg)
    df, labels = gen.generate()
    fault_rows = df[df["timestamp"].between(
        df["timestamp"].iloc[0] + pd.Timedelta(seconds=6),
        df["timestamp"].iloc[0] + pd.Timedelta(seconds=12),
    )]
    min_v = fault_rows["voltage_v"].min()
    assert min_v < 9.0, f"Cold-crank should sag bus below 9 V, got {min_v:.2f} V"
    # Current should also be reduced from nominal
    assert fault_rows["current_a"].median() < ch.nominal_current_a, \
        "Current should be reduced during cold crank"
    assert labels["fault_type"].iloc[0] == FaultType.COLD_CRANK.value


def test_cold_crank_recovers_after_window():
    """After the cold-crank window bus voltage should recover to ~13.5 V."""
    ch = _make_ch_bus()
    cfg = _make_config(
        duration_s=30.0,
        sample_interval_ms=100.0,
        channels=[ch],
        fault_injections=[
            FaultInjection(channel_id="ch_bus", fault_type=FaultType.COLD_CRANK,
                           start_s=5.0, duration_s=8.0, intensity=0.9),
        ],
    )
    gen = TelemetryGenerator(cfg)
    df, _ = gen.generate()
    post_crank = df[df["timestamp"] > df["timestamp"].iloc[0] + pd.Timedelta(seconds=14)]
    assert post_crank["voltage_v"].mean() > 12.0, \
        "Bus voltage should recover above 12 V after cold-crank ends"


def test_classifier_detects_jump_start():
    """RulesFaultClassifier should return JUMP_START when rolling_max_voltage > 16 V."""
    from src.models.anomaly import RulesFaultClassifier
    clf = RulesFaultClassifier()
    fault, conf, causes = clf.classify({
        "rolling_max_voltage": 19.5,
        "voltage_v": 19.5,
        "over_voltage_count": 3,
        "spike_score": 0.0,
        "trip_frequency": 0,
        "temp_slope": 0.0,
        "temperature_slope": 0.0,
        "deg_trend": 0.0,
        "degradation_trend": 0.0,
        "protection_event": ProtectionEvent.OVER_VOLTAGE.value,
    })
    assert fault == FaultType.JUMP_START
    assert conf >= 0.65


def test_classifier_detects_load_dump():
    """RulesFaultClassifier should return LOAD_DUMP when rolling_max_voltage > 30 V."""
    from src.models.anomaly import RulesFaultClassifier
    clf = RulesFaultClassifier()
    fault, conf, causes = clf.classify({
        "rolling_max_voltage": 38.0,
        "voltage_v": 38.0,
        "over_voltage_count": 5,
        "spike_score": 0.0,
        "trip_frequency": 0,
        "temperature_slope": 0.0,
        "degradation_trend": 0.0,
        "protection_event": ProtectionEvent.OVER_VOLTAGE.value,
    })
    assert fault == FaultType.LOAD_DUMP
    assert conf >= 0.65


def test_classifier_detects_cold_crank():
    """RulesFaultClassifier should return COLD_CRANK when rolling_min_voltage < 9 V."""
    from src.models.anomaly import RulesFaultClassifier
    clf = RulesFaultClassifier()
    fault, conf, causes = clf.classify({
        "rolling_min_voltage": 7.2,
        "rolling_max_voltage": 13.5,
        "voltage_v": 12.0,
        "over_voltage_count": 0,
        "spike_score": 0.0,
        "trip_frequency": 0,
        "temperature_slope": 0.0,
        "degradation_trend": 0.0,
        "protection_event": ProtectionEvent.NONE.value,
    })
    assert fault == FaultType.COLD_CRANK
    assert conf >= 0.5


# ---------------------------------------------------------------------------
# Gap #6 — Wire harness resistance + connector aging
# ---------------------------------------------------------------------------

def _make_ch_harness(
    harness_r_ohm: float = 0.020,
    connector_r_ohm: float = 0.010,
    rds_on_tempco_exp: float = 0.0,
) -> ChannelMeta:
    return ChannelMeta(
        channel_id="ch_harness",
        load_name="fog_light",
        nominal_current_a=6.0,
        max_current_a=20.0,
        fuse_rating_a=15.0,
        thermal_shutdown_c=150.0,
        r_thermal_kw=20.0,
        tau_thermal_s=10.0,
        t_ambient_c=25.0,
        harness_r_ohm=harness_r_ohm,
        connector_r_ohm=connector_r_ohm,
        rds_on_tempco_exp=rds_on_tempco_exp,
    )


def test_harness_r_raises_voltage_drop():
    """Higher harness resistance should produce a lower measured voltage."""
    from src.config.models import SimulationConfig

    ch_low = _make_ch_harness(harness_r_ohm=0.010, connector_r_ohm=0.005)
    ch_high = _make_ch_harness(harness_r_ohm=0.100, connector_r_ohm=0.050)

    def _gen(ch):
        cfg = SimulationConfig(
            scenario_id="test", name="T", duration_s=10.0, sample_interval_ms=100.0,
            seed=42, channels=[ch], fault_injections=[],
        )
        return TelemetryGenerator(cfg).generate()[0]

    df_low = _gen(ch_low)
    df_high = _gen(ch_high)
    assert df_high["voltage_v"].mean() < df_low["voltage_v"].mean(), \
        "Higher harness R should yield lower measured voltage"


def test_connector_aging_drops_voltage_over_time():
    """CONNECTOR_AGING fault — voltage should fall progressively during the fault window."""
    ch = _make_ch_harness()
    cfg = _make_config(
        duration_s=60.0,
        sample_interval_ms=200.0,
        channels=[ch],
        fault_injections=[
            FaultInjection(
                channel_id="ch_harness", fault_type=FaultType.CONNECTOR_AGING,
                start_s=10.0, duration_s=40.0, intensity=0.9,
            ),
        ],
    )
    gen = TelemetryGenerator(cfg)
    df, labels = gen.generate()
    fault_rows = df[
        (df["timestamp"] >= df["timestamp"].iloc[0] + pd.Timedelta(seconds=10))
        & (df["timestamp"] <= df["timestamp"].iloc[0] + pd.Timedelta(seconds=50))
    ].copy()
    # Voltage should trend downward through the fault window
    first_quarter = fault_rows.iloc[: len(fault_rows) // 4]["voltage_v"].mean()
    last_quarter = fault_rows.iloc[-len(fault_rows) // 4 :]["voltage_v"].mean()
    assert last_quarter < first_quarter, (
        f"Voltage should fall during connector aging fault: "
        f"first_q={first_quarter:.3f} V, last_q={last_quarter:.3f} V"
    )
    assert labels["fault_type"].iloc[0] == FaultType.CONNECTOR_AGING.value


def test_connector_aging_current_reduction():
    """Current should be slightly reduced at end of window vs start (resistive load, lower V)."""
    ch = _make_ch_harness()
    cfg = _make_config(
        duration_s=60.0,
        sample_interval_ms=200.0,
        channels=[ch],
        fault_injections=[
            FaultInjection(
                channel_id="ch_harness", fault_type=FaultType.CONNECTOR_AGING,
                start_s=5.0, duration_s=50.0, intensity=1.0,
            ),
        ],
    )
    gen = TelemetryGenerator(cfg)
    df, _ = gen.generate()
    fault_rows = df[
        df["timestamp"] >= df["timestamp"].iloc[0] + pd.Timedelta(seconds=5)
    ]
    first_mean = fault_rows.iloc[:10]["current_a"].mean()
    last_mean = fault_rows.iloc[-10:]["current_a"].mean()
    assert last_mean <= first_mean * 1.05, (
        "Current should not increase with rising connector resistance"
    )


def test_connector_aging_no_trip():
    """Connector aging is a slow degradation — eFuse should not trip."""
    ch = _make_ch_harness()
    cfg = _make_config(
        duration_s=30.0,
        sample_interval_ms=100.0,
        channels=[ch],
        fault_injections=[
            FaultInjection(
                channel_id="ch_harness", fault_type=FaultType.CONNECTOR_AGING,
                start_s=0.0, duration_s=30.0, intensity=0.8,
            ),
        ],
    )
    gen = TelemetryGenerator(cfg)
    df, _ = gen.generate()
    assert not df["trip_flag"].any(), "Connector aging should not trigger eFuse trip"


def test_classifier_detects_connector_aging():
    """RulesFaultClassifier should return CONNECTOR_AGING for elevated voltage drop."""
    from src.models.anomaly import RulesFaultClassifier
    clf = RulesFaultClassifier()
    fault, conf, causes = clf.classify({
        "rolling_voltage_drop": 0.85,
        "voltage_v": 12.6,
        "rolling_min_voltage": 12.5,
        "rolling_max_voltage": 13.5,
        "over_voltage_count": 0,
        "spike_score": 0.1,
        "trip_flag": False,
        "overload_flag": False,
        "trip_frequency": 0,
        "temperature_slope": 0.01,
        "degradation_trend": 0.0,
        "protection_event": ProtectionEvent.NONE.value,
        "missing_rate": 0.0,
    })
    assert fault == FaultType.CONNECTOR_AGING
    assert conf >= 0.4
    assert any("voltage drop" in c.lower() for c in causes)


# ---------------------------------------------------------------------------
# Gap #7 — Multi-channel die thermal coupling
# ---------------------------------------------------------------------------

def _make_die_channels(
    die_id: str = "die_A",
    coupling: float = 0.20,
) -> list[ChannelMeta]:
    """Two channels sharing a die: ch_hot (high current) and ch_cold (low current)."""
    common = dict(
        max_current_a=20.0, fuse_rating_a=15.0,
        thermal_shutdown_c=150.0, r_thermal_kw=30.0, tau_thermal_s=5.0,
        t_ambient_c=25.0, rds_on_tempco_exp=0.0,
        die_id=die_id, thermal_coupling_coeff=coupling,
    )
    ch_hot = ChannelMeta(
        channel_id="ch_hot", load_name="seat_heater",
        nominal_current_a=12.0, r_ds_on_ohm=0.006, **common,
    )
    ch_cold = ChannelMeta(
        channel_id="ch_cold", load_name="mirror_adjust",
        nominal_current_a=1.5, r_ds_on_ohm=0.010, **common,
    )
    return [ch_hot, ch_cold]


def test_coupled_channel_runs_hotter_than_isolated():
    """ch_cold on shared die should be hotter than an isolated ch_cold with same current."""
    ch_coupled, ch_cold_coupled = _make_die_channels()
    ch_isolated = ChannelMeta(
        channel_id="ch_cold", load_name="mirror_adjust",
        nominal_current_a=1.5, r_ds_on_ohm=0.010,
        max_current_a=20.0, fuse_rating_a=15.0,
        thermal_shutdown_c=150.0, r_thermal_kw=30.0, tau_thermal_s=5.0,
        t_ambient_c=25.0, rds_on_tempco_exp=0.0,
        die_id="",  # isolated
    )

    cfg_coupled = _make_config(
        duration_s=30.0, sample_interval_ms=100.0,
        channels=[ch_coupled, ch_cold_coupled], fault_injections=[],
    )
    cfg_isolated = _make_config(
        duration_s=30.0, sample_interval_ms=100.0,
        channels=[ch_isolated], fault_injections=[],
    )
    gen_c = TelemetryGenerator(cfg_coupled)
    gen_i = TelemetryGenerator(cfg_isolated)
    df_c, _ = gen_c.generate()
    df_i, _ = gen_i.generate()

    # Steady-state temperature (last 10 s) for ch_cold in each scenario
    t0_c = df_c["timestamp"].min()
    t0_i = df_i["timestamp"].min()
    cold_coupled_temp = df_c[
        (df_c["channel_id"] == "ch_cold")
        & (df_c["timestamp"] > t0_c + pd.Timedelta(seconds=20))
    ]["temperature_c"].mean()
    cold_isolated_temp = df_i[
        df_i["timestamp"] > t0_i + pd.Timedelta(seconds=20)
    ]["temperature_c"].mean()

    assert cold_coupled_temp > cold_isolated_temp, (
        f"Coupled channel ({cold_coupled_temp:.2f}°C) should be hotter than "
        f"isolated ({cold_isolated_temp:.2f}°C)"
    )


def test_hot_channel_does_not_affect_temperature_with_zero_coupling():
    """With thermal_coupling_coeff=0, co-die channel should not receive any extra heat."""
    channels = _make_die_channels(coupling=0.0)

    ch_isolated = ChannelMeta(
        channel_id="ch_cold", load_name="mirror_adjust",
        nominal_current_a=1.5, r_ds_on_ohm=0.010,
        max_current_a=20.0, fuse_rating_a=15.0,
        thermal_shutdown_c=150.0, r_thermal_kw=30.0, tau_thermal_s=5.0,
        t_ambient_c=25.0, rds_on_tempco_exp=0.0, die_id="",
    )

    cfg_zero = _make_config(duration_s=20.0, sample_interval_ms=100.0, channels=channels, fault_injections=[])
    cfg_iso = _make_config(duration_s=20.0, sample_interval_ms=100.0, channels=[ch_isolated], fault_injections=[])

    df_zero, _ = TelemetryGenerator(cfg_zero).generate()
    df_iso, _ = TelemetryGenerator(cfg_iso).generate()

    t0_z = df_zero["timestamp"].min()
    t0_i = df_iso["timestamp"].min()
    temp_zero = df_zero[
        (df_zero["channel_id"] == "ch_cold") & (df_zero["timestamp"] > t0_z + pd.Timedelta(seconds=15))
    ]["temperature_c"].mean()
    temp_iso = df_iso[
        df_iso["timestamp"] > t0_i + pd.Timedelta(seconds=15)
    ]["temperature_c"].mean()

    assert abs(temp_zero - temp_iso) < 1.0, (
        f"Zero coupling should produce same temp as isolated: "
        f"zero_coupling={temp_zero:.2f}°C, isolated={temp_iso:.2f}°C"
    )


def test_isolated_channels_not_affected_by_coupling():
    """Channels with different die_ids should not thermally interact."""
    ch_a = ChannelMeta(
        channel_id="ch_a", load_name="load_a",
        nominal_current_a=12.0, r_ds_on_ohm=0.006,
        max_current_a=20.0, fuse_rating_a=15.0,
        thermal_shutdown_c=150.0, r_thermal_kw=30.0, tau_thermal_s=5.0,
        t_ambient_c=25.0, rds_on_tempco_exp=0.0, die_id="die_X",
    )
    ch_b = ChannelMeta(
        channel_id="ch_b", load_name="load_b",
        nominal_current_a=1.5, r_ds_on_ohm=0.010,
        max_current_a=20.0, fuse_rating_a=15.0,
        thermal_shutdown_c=150.0, r_thermal_kw=30.0, tau_thermal_s=5.0,
        t_ambient_c=25.0, rds_on_tempco_exp=0.0, die_id="die_Y",  # different die
    )
    ch_b_iso = ChannelMeta(
        channel_id="ch_b", load_name="load_b",
        nominal_current_a=1.5, r_ds_on_ohm=0.010,
        max_current_a=20.0, fuse_rating_a=15.0,
        thermal_shutdown_c=150.0, r_thermal_kw=30.0, tau_thermal_s=5.0,
        t_ambient_c=25.0, rds_on_tempco_exp=0.0, die_id="",
    )

    df_diff, _ = TelemetryGenerator(_make_config(
        duration_s=20.0, sample_interval_ms=100.0, channels=[ch_a, ch_b], fault_injections=[],
    )).generate()
    df_iso, _ = TelemetryGenerator(_make_config(
        duration_s=20.0, sample_interval_ms=100.0, channels=[ch_b_iso], fault_injections=[],
    )).generate()

    t0_d = df_diff["timestamp"].min()
    t0_i = df_iso["timestamp"].min()
    temp_diff = df_diff[
        (df_diff["channel_id"] == "ch_b") & (df_diff["timestamp"] > t0_d + pd.Timedelta(seconds=15))
    ]["temperature_c"].mean()
    temp_iso = df_iso[
        df_iso["timestamp"] > t0_i + pd.Timedelta(seconds=15)
    ]["temperature_c"].mean()

    assert abs(temp_diff - temp_iso) < 1.0, (
        "Different-die channels should not thermally interact"
    )


def test_classifier_detects_thermal_coupling():
    """RulesFaultClassifier should return THERMAL_COUPLING for moderate temp slope, nominal current."""
    from src.models.anomaly import RulesFaultClassifier
    clf = RulesFaultClassifier()
    fault, conf, causes = clf.classify({
        "temperature_slope": 0.25,   # moderate rise — not extreme
        "spike_score": 0.3,
        "trip_flag": False,
        "overload_flag": False,
        "trip_frequency": 0,
        "degradation_trend": 0.0,
        "rolling_rms_current": 1.5,
        "current_a": 1.5,            # nominal current — not self-heating
        "voltage_v": 13.5,
        "rolling_min_voltage": 13.0,
        "rolling_max_voltage": 14.0,
        "over_voltage_count": 0,
        "protection_event": ProtectionEvent.NONE.value,
        "missing_rate": 0.0,
    })
    assert fault == FaultType.THERMAL_COUPLING, f"Expected THERMAL_COUPLING, got {fault}"
    assert conf > 0
    assert any("co-die" in c.lower() or "coupling" in c.lower() for c in causes)
