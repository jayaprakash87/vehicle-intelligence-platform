"""Tests for eFuse catalog, vehicle topology, and fleet-scale channel generation."""

from __future__ import annotations

import pytest

from src.config.catalog import (
    EFUSE_CATALOG,
    build_channels,
    get_profile,
    sedan_topology,
)
from src.config.models import SimulationConfig, load_config
from src.schemas.telemetry import (
    ChannelMeta,
    DriverType,
    EFuseFamily,
    PowerClass,
    SourceProtocol,
    ZoneController,
)
from src.simulation.generator import TelemetryGenerator
from src.ingestion.normalizer import Normalizer
from src.features.engine import FeatureEngine
from src.config.models import FeatureConfig, NormalizerConfig


# ---------------------------------------------------------------------------
# eFuse catalog
# ---------------------------------------------------------------------------

class TestEFuseCatalog:
    def test_all_families_have_profiles(self):
        """Every EFuseFamily enum value must have a catalog entry."""
        for fam in EFuseFamily:
            assert fam in EFUSE_CATALOG, f"Missing catalog entry for {fam}"

    def test_profiles_have_sensible_values(self):
        for fam, profile in EFUSE_CATALOG.items():
            assert profile.nominal_current_a > 0
            assert profile.max_current_a > profile.nominal_current_a
            assert profile.fuse_rating_a > 0
            assert profile.r_ds_on_ohm > 0
            assert profile.r_thermal_kw > 0
            assert profile.tau_thermal_s > 0

    def test_higher_rated_efuse_has_lower_rds_on(self):
        """Higher current eFuses should have lower on-resistance (bigger die)."""
        hs_2a = get_profile(EFuseFamily.HS_2A)
        hs_50a = get_profile(EFuseFamily.HS_50A)
        assert hs_50a.r_ds_on_ohm < hs_2a.r_ds_on_ohm

    def test_get_profile_returns_correct_family(self):
        p = get_profile(EFuseFamily.HS_10A)
        assert p.efuse_family == EFuseFamily.HS_10A
        assert p.nominal_current_a == 6.0

    def test_all_profiles_have_ic_identity(self):
        """Every catalog entry should reference a real IC part number."""
        for fam, profile in EFUSE_CATALOG.items():
            assert profile.ic_part_number != "", f"{fam}: missing ic_part_number"
            assert profile.manufacturer != "", f"{fam}: missing manufacturer"

    def test_known_manufacturers(self):
        manufacturers = {p.manufacturer for p in EFUSE_CATALOG.values()}
        assert manufacturers <= {"Infineon", "STMicroelectronics"}, f"Unexpected: {manufacturers}"

    def test_h_bridge_ic_in_catalog(self):
        """VNH9045 should be referenced for the LS_15A family."""
        ls_15a = get_profile(EFuseFamily.LS_15A)
        assert ls_15a.ic_part_number == "VNH9045"


# ---------------------------------------------------------------------------
# Vehicle topology
# ---------------------------------------------------------------------------

class TestSedanTopology:
    def test_sedan_has_52_channels(self):
        zones, specs = sedan_topology()
        assert len(specs) == 52

    def test_sedan_has_4_zones(self):
        zones, specs = sedan_topology()
        assert len(zones) == 4
        zone_ids = {z.zone_id for z in zones}
        assert zone_ids == {"zone_body", "zone_front", "zone_rear", "zone_underhood"}

    def test_all_specs_have_required_fields(self):
        zones, specs = sedan_topology()
        for s in specs:
            assert "channel_id" in s
            assert "efuse_family" in s
            assert "zone_id" in s
            assert "load_name" in s

    def test_zone_channel_counts(self):
        zones, specs = sedan_topology()
        zone_counts = {}
        for s in specs:
            z = s["zone_id"]
            zone_counts[z] = zone_counts.get(z, 0) + 1
        assert zone_counts["zone_body"] == 16
        assert zone_counts["zone_front"] == 14
        assert zone_counts["zone_rear"] == 10
        assert zone_counts["zone_underhood"] == 12

    def test_unique_channel_ids(self):
        zones, specs = sedan_topology()
        ids = [s["channel_id"] for s in specs]
        assert len(ids) == len(set(ids)), "Duplicate channel IDs found"


# ---------------------------------------------------------------------------
# Channel factory (build_channels)
# ---------------------------------------------------------------------------

class TestBuildChannels:
    def test_builds_full_channel_list(self):
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        assert len(channels) == 52
        assert all(isinstance(ch, ChannelMeta) for ch in channels)

    def test_inherits_catalog_defaults(self):
        """Channel without overrides should get catalog profile values."""
        zones = [ZoneController(zone_id="z1", name="Test")]
        specs = [{"channel_id": "ch_test", "zone_id": "z1", "efuse_family": "hs_10a", "load_name": "test_load"}]
        channels = build_channels(zones, specs)
        ch = channels[0]
        profile = get_profile(EFuseFamily.HS_10A)
        assert ch.nominal_current_a == profile.nominal_current_a
        assert ch.max_current_a == profile.max_current_a
        assert ch.r_ds_on_ohm == profile.r_ds_on_ohm
        assert ch.efuse_family == EFuseFamily.HS_10A

    def test_spec_overrides_catalog(self):
        """Explicit per-channel values override catalog defaults."""
        zones = [ZoneController(zone_id="z1", name="Test")]
        specs = [{
            "channel_id": "ch_test",
            "zone_id": "z1",
            "efuse_family": "hs_10a",
            "load_name": "test_load",
            "nominal_current_a": 7.5,  # override catalog's 6.0
        }]
        channels = build_channels(zones, specs)
        assert channels[0].nominal_current_a == 7.5

    def test_inherits_zone_protocol(self):
        """Channel should inherit source_protocol from its zone controller."""
        zones = [ZoneController(zone_id="z1", name="XCP Zone", bus_interface=SourceProtocol.XCP)]
        specs = [{"channel_id": "ch_test", "zone_id": "z1", "efuse_family": "hs_5a", "load_name": "test"}]
        channels = build_channels(zones, specs)
        assert channels[0].source_protocol == SourceProtocol.XCP

    def test_connected_loads_preserved(self):
        zones = [ZoneController(zone_id="z1")]
        specs = [{
            "channel_id": "ch_test",
            "zone_id": "z1",
            "efuse_family": "hs_30a",
            "load_name": "seat_heater",
            "connected_loads": ["seat_heater_left", "lumbar_support"],
        }]
        channels = build_channels(zones, specs)
        assert channels[0].connected_loads == ["seat_heater_left", "lumbar_support"]

    def test_no_zone_assignment(self):
        """Channels without zone_id should still build fine."""
        zones = []
        specs = [{"channel_id": "ch_test", "efuse_family": "hs_2a", "load_name": "led"}]
        channels = build_channels(zones, specs)
        assert len(channels) == 1
        assert channels[0].zone_id == ""

    def test_propagates_thermal_shutdown_c(self):
        """build_channels should propagate thermal_shutdown_c from catalog."""
        zones = [ZoneController(zone_id="z1", name="Test")]
        specs = [{"channel_id": "ch_test", "zone_id": "z1", "efuse_family": "hs_10a", "load_name": "test"}]
        channels = build_channels(zones, specs)
        ch = channels[0]
        profile = get_profile(EFuseFamily.HS_10A)
        assert ch.thermal_shutdown_c == profile.thermal_shutdown_c


# ---------------------------------------------------------------------------
# Config loading with topology
# ---------------------------------------------------------------------------

class TestTopologyConfig:
    def test_load_sedan_config(self):
        from pathlib import Path
        cfg_path = Path(__file__).parent.parent / "configs" / "sedan_52ch.yaml"
        if not cfg_path.exists():
            pytest.skip("sedan_52ch.yaml not found")
        cfg = load_config(str(cfg_path))
        assert len(cfg.simulation.channels) == 52
        assert len(cfg.simulation.zones) == 4

    def test_unknown_topology_raises(self):
        cfg = SimulationConfig(vehicle_topology="suv")
        from src.config.models import PlatformConfig, _resolve_topology
        platform = PlatformConfig(simulation=cfg)
        with pytest.raises(ValueError, match="Unknown vehicle_topology"):
            _resolve_topology(platform)


# ---------------------------------------------------------------------------
# Fleet-scale generation (52 channels)
# ---------------------------------------------------------------------------

class TestFleetScaleGeneration:
    def test_generate_52_channels(self):
        """Generate telemetry for all 52 sedan channels."""
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        cfg = SimulationConfig(
            duration_s=5.0,
            sample_interval_ms=100.0,
            channels=channels,
            seed=42,
        )
        gen = TelemetryGenerator(cfg)
        telem_df, _ = gen.generate()

        unique_channels = telem_df["channel_id"].nunique()
        assert unique_channels == 52
        # 5s / 100ms = 50 rows per channel × 52 channels = 2600
        assert len(telem_df) == 52 * 50

    def test_52ch_normalize_and_features(self):
        """Full pipeline on 52 channels: generate → normalize → features."""
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        cfg = SimulationConfig(
            duration_s=2.0,
            sample_interval_ms=100.0,
            channels=channels,
            seed=42,
        )
        gen = TelemetryGenerator(cfg)
        telem_df, _ = gen.generate()

        norm = Normalizer(NormalizerConfig())
        norm_df = norm.normalize(telem_df)
        assert len(norm_df) == len(telem_df)

        engine = FeatureEngine(FeatureConfig(window_duration_s=1.0, min_duration_s=0.5))
        feat_df = engine.compute(norm_df)
        assert "rolling_rms_current" in feat_df.columns
        assert feat_df["channel_id"].nunique() == 52

    def test_52ch_with_faults(self):
        """Generate with faults injected across multiple zones."""
        from src.schemas.telemetry import FaultInjection, FaultType
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        faults = [
            FaultInjection(channel_id="ch_017", fault_type=FaultType.OVERLOAD_SPIKE, start_s=1.0, duration_s=1.0, intensity=0.8),
            FaultInjection(channel_id="ch_041", fault_type=FaultType.VOLTAGE_SAG, start_s=2.0, duration_s=1.0, intensity=0.7),
        ]
        cfg = SimulationConfig(
            duration_s=5.0,
            sample_interval_ms=100.0,
            channels=channels,
            fault_injections=faults,
            seed=42,
        )
        gen = TelemetryGenerator(cfg)
        telem_df, labels_df = gen.generate()
        assert len(labels_df) > 0
        assert set(labels_df["channel_id"].unique()) == {"ch_017", "ch_041"}


# ---------------------------------------------------------------------------
# IO attributes — system hierarchy, driver type, power class, PWM
# ---------------------------------------------------------------------------

class TestIOAttributes:
    """Verify the sedan topology populates the new IO-level fields."""

    def test_all_channels_have_system_cluster(self):
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        for ch in channels:
            assert ch.system_cluster != "", f"{ch.channel_id} ({ch.load_name}) missing system_cluster"

    def test_all_channels_have_system_name(self):
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        for ch in channels:
            assert ch.system_name != "", f"{ch.channel_id} ({ch.load_name}) missing system_name"

    def test_known_system_clusters(self):
        """All clusters should be from a known set."""
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        clusters = {ch.system_cluster for ch in channels}
        expected = {
            "interior_lighting", "exterior_lighting", "body_comfort",
            "driver_assist", "adas", "powertrain",
        }
        assert clusters == expected, f"Unexpected clusters: {clusters - expected}"

    def test_always_on_channels(self):
        """DRLs, tail lights, horn, etc. should be always_on (KL30)."""
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        always_on = [ch for ch in channels if ch.power_class == PowerClass.ALWAYS_ON]
        always_on_names = {ch.load_name for ch in always_on}
        # DRLs and tail lights are safety-critical = always on
        assert "drl_left" in always_on_names
        assert "tail_light_left" in always_on_names
        assert "horn" in always_on_names

    def test_starter_channel(self):
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        starter = [ch for ch in channels if ch.load_name == "starter_relay"]
        assert len(starter) == 1
        assert starter[0].power_class == PowerClass.START

    def test_h_bridge_motors(self):
        """Bidirectional motors (windows, mirrors, locks) should use H-bridge."""
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        windows = [ch for ch in channels if "window" in ch.load_name]
        for ch in windows:
            assert ch.driver_type == DriverType.H_BRIDGE, f"{ch.load_name} should be H-bridge"

    def test_low_side_driver(self):
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        ls = [ch for ch in channels if ch.driver_type == DriverType.LOW_SIDE]
        assert any(ch.load_name == "trunk_latch" for ch in ls)

    def test_pwm_capable_lights(self):
        """Dimmable lights should be PWM-capable."""
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        dome = [ch for ch in channels if ch.load_name == "dome_light"][0]
        assert dome.pwm_capable is True
        ambient = [ch for ch in channels if ch.load_name == "ambient_led_driver"][0]
        assert ambient.pwm_capable is True

    def test_non_pwm_defaults(self):
        """Simple on/off loads shouldn't be PWM-capable."""
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        horn = [ch for ch in channels if ch.load_name == "horn"][0]
        assert horn.pwm_capable is False

    def test_capacitive_loads(self):
        """ECU/sensor power supplies should be capacitive load type."""
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        adas = [ch for ch in channels if ch.load_name == "adas_camera_power"][0]
        assert adas.load_type == "capacitive"
        rain = [ch for ch in channels if ch.load_name == "rain_sensor"][0]
        assert rain.load_type == "capacitive"

    def test_default_driver_type_is_high_side(self):
        """Most channels should default to high-side driver."""
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        hs_count = sum(1 for ch in channels if ch.driver_type == DriverType.HIGH_SIDE)
        # Majority should be high-side (all except ~10 H-bridge/LS/half-bridge)
        assert hs_count > 30

    def test_default_power_class_is_ignition(self):
        """Most channels should be ignition-switched (KL15)."""
        zones, specs = sedan_topology()
        channels = build_channels(zones, specs)
        ign_count = sum(1 for ch in channels if ch.power_class == PowerClass.IGNITION)
        assert ign_count > 35


# ---------------------------------------------------------------------------
# Schema additions
# ---------------------------------------------------------------------------

class TestSchemaAdditions:
    def test_efuse_family_enum_values(self):
        assert EFuseFamily.HS_2A.value == "hs_2a"
        assert EFuseFamily.HS_50A.value == "hs_50a"
        assert EFuseFamily.LS_5A.value == "ls_5a"

    def test_zone_controller_defaults(self):
        zc = ZoneController(zone_id="test")
        assert zc.location == "body"
        assert zc.bus_interface == SourceProtocol.CAN
        assert zc.cdd_read_cycle_ms == 10.0

    def test_channel_meta_new_fields_backward_compatible(self):
        """ChannelMeta with no new fields should still work."""
        ch = ChannelMeta(channel_id="old_style", nominal_current_a=5.0, max_current_a=10.0)
        assert ch.efuse_family == EFuseFamily.HS_15A
        assert ch.zone_id == ""
        assert ch.connected_loads == []

    def test_channel_meta_with_new_fields(self):
        ch = ChannelMeta(
            channel_id="new_style",
            efuse_family=EFuseFamily.HS_30A,
            zone_id="zone_body",
            connected_loads=["seat_heater", "lumbar"],
            nominal_current_a=20.0,
            max_current_a=40.0,
        )
        assert ch.efuse_family == EFuseFamily.HS_30A
        assert ch.zone_id == "zone_body"
        assert len(ch.connected_loads) == 2
