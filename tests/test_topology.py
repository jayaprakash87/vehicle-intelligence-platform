"""Tests for eFuse catalog, vehicle topology, and fleet-scale channel generation."""

from __future__ import annotations

import pytest

from src.config.catalog import (
    EFUSE_CATALOG,
    build_channels,
    example_topology,
    get_profile,
)
from src.config.models import (
    FeatureConfig,
    NormalizerConfig,
    PlatformConfig,
    SimulationConfig,
    _resolve_topology,
    load_config,
)
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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def topo_zones_specs():
    return example_topology()


@pytest.fixture(scope="module")
def topo_channels(topo_zones_specs):
    zones, specs = topo_zones_specs
    return build_channels(zones, specs)


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
        low = get_profile(EFuseFamily.INF_HS_2A)
        high = get_profile(EFuseFamily.INF_HS_100A)
        assert high.r_ds_on_ohm < low.r_ds_on_ohm

    def test_get_profile_returns_correct_family(self):
        p = get_profile(EFuseFamily.INF_HS_11A)
        assert p.efuse_family == EFuseFamily.INF_HS_11A
        assert p.nominal_current_a == 7.0

    def test_all_profiles_have_ic_identity(self):
        """Every catalog entry should reference a real IC part number."""
        for fam, profile in EFUSE_CATALOG.items():
            assert profile.ic_part_number != "", f"{fam}: missing ic_part_number"
            assert profile.manufacturer != "", f"{fam}: missing manufacturer"

    def test_known_manufacturers(self):
        manufacturers = {p.manufacturer for p in EFUSE_CATALOG.values()}
        assert manufacturers <= {"Infineon", "STMicroelectronics", "custom"}, (
            f"Unexpected: {manufacturers}"
        )

    def test_h_bridge_ic_in_catalog(self):
        """VNH9045AQTR should be referenced for the ST_HB_30A family."""
        hb = get_profile(EFuseFamily.ST_HB_30A)
        assert hb.ic_part_number == "VNH9045AQTR"


# ---------------------------------------------------------------------------
# Channel factory (build_channels)
# ---------------------------------------------------------------------------


class TestBuildChannels:
    def test_builds_full_channel_list(self):
        zones, specs = example_topology()
        channels = build_channels(zones, specs)
        assert len(channels) == 65
        assert all(isinstance(ch, ChannelMeta) for ch in channels)

    def test_inherits_catalog_defaults(self):
        """Channel without overrides should get catalog profile values."""
        zones = [ZoneController(zone_id="z1", name="Test")]
        specs = [
            {
                "channel_id": "ch_test",
                "zone_id": "z1",
                "efuse_family": "inf_hs_11a",
                "load_name": "test_load",
            }
        ]
        channels = build_channels(zones, specs)
        ch = channels[0]
        profile = get_profile(EFuseFamily.INF_HS_11A)
        assert ch.nominal_current_a == profile.nominal_current_a
        assert ch.max_current_a == profile.max_current_a
        assert ch.r_ds_on_ohm == profile.r_ds_on_ohm
        assert ch.efuse_family == EFuseFamily.INF_HS_11A

    def test_spec_overrides_catalog(self):
        """Explicit per-channel values override catalog defaults."""
        zones = [ZoneController(zone_id="z1", name="Test")]
        specs = [
            {
                "channel_id": "ch_test",
                "zone_id": "z1",
                "efuse_family": "inf_hs_11a",
                "load_name": "test_load",
                "nominal_current_a": 8.5,  # override catalog's 7.0
            }
        ]
        channels = build_channels(zones, specs)
        assert channels[0].nominal_current_a == 8.5

    def test_inherits_zone_protocol(self):
        """Channel should inherit source_protocol from its zone controller."""
        zones = [ZoneController(zone_id="z1", name="XCP Zone", bus_interface=SourceProtocol.XCP)]
        specs = [
            {
                "channel_id": "ch_test",
                "zone_id": "z1",
                "efuse_family": "inf_hs_5a",
                "load_name": "test",
            }
        ]
        channels = build_channels(zones, specs)
        assert channels[0].source_protocol == SourceProtocol.XCP

    def test_connected_loads_preserved(self):
        zones = [ZoneController(zone_id="z1")]
        specs = [
            {
                "channel_id": "ch_test",
                "zone_id": "z1",
                "efuse_family": "st_hs_30a",
                "load_name": "seat_heater",
                "connected_loads": ["seat_heater_left", "lumbar_support"],
            }
        ]
        channels = build_channels(zones, specs)
        assert channels[0].connected_loads == ["seat_heater_left", "lumbar_support"]

    def test_no_zone_assignment(self):
        """Channels without zone_id should still build fine."""
        zones = []
        specs = [{"channel_id": "ch_test", "efuse_family": "inf_hs_2a", "load_name": "led"}]
        channels = build_channels(zones, specs)
        assert len(channels) == 1
        assert channels[0].zone_id == ""

    def test_propagates_thermal_shutdown_c(self):
        """build_channels should propagate thermal_shutdown_c from catalog."""
        zones = [ZoneController(zone_id="z1", name="Test")]
        specs = [
            {
                "channel_id": "ch_test",
                "zone_id": "z1",
                "efuse_family": "inf_hs_11a",
                "load_name": "test",
            }
        ]
        channels = build_channels(zones, specs)
        ch = channels[0]
        profile = get_profile(EFuseFamily.INF_HS_11A)
        assert ch.thermal_shutdown_c == profile.thermal_shutdown_c


# ---------------------------------------------------------------------------
# Config loading with topology
# ---------------------------------------------------------------------------


class TestTopologyConfig:
    def test_load_example_config(self):
        from pathlib import Path

        cfg_path = Path(__file__).parent.parent / "configs" / "example_65ch.yaml"
        if not cfg_path.exists():
            pytest.skip("example_65ch.yaml not found")
        cfg = load_config(str(cfg_path))
        assert len(cfg.simulation.channels) == 65
        assert len(cfg.simulation.zones) == 4

    def test_no_topology_flag_keeps_default_channels(self):
        """use_example_topology=False with no channel_specs leaves channels as default."""
        cfg = SimulationConfig(use_example_topology=False)
        platform = PlatformConfig(simulation=cfg)
        _resolve_topology(platform)  # no-op — uses default explicit channels
        assert len(platform.simulation.channels) == 3  # default 3-channel list


# ---------------------------------------------------------------------------
# Example topology — generic 4-zone architecture
# ---------------------------------------------------------------------------


class TestExampleTopology:
    def test_example_has_65_channels(self, topo_zones_specs):
        zones, specs = topo_zones_specs
        assert len(specs) == 65

    def test_example_has_4_zone_controllers(self, topo_zones_specs):
        zones, specs = topo_zones_specs
        assert len(zones) == 4
        zone_ids = {z.zone_id for z in zones}
        assert zone_ids == {"zone_rear", "zone_body", "zone_front", "zone_central"}

    def test_example_zone_channel_counts(self, topo_zones_specs):
        zones, specs = topo_zones_specs
        zone_counts: dict[str, int] = {}
        for s in specs:
            z = s["zone_id"]
            zone_counts[z] = zone_counts.get(z, 0) + 1
        assert zone_counts["zone_rear"] == 25
        assert zone_counts["zone_body"] == 15
        assert zone_counts["zone_front"] == 15
        assert zone_counts["zone_central"] == 10

    def test_example_config_loading(self):
        """Example topology can be loaded via _resolve_topology."""
        cfg = SimulationConfig(use_example_topology=True)
        platform = PlatformConfig(simulation=cfg)
        _resolve_topology(platform)
        assert len(platform.simulation.channels) == 65
        assert len(platform.simulation.zones) == 4

    def test_example_unique_channel_ids(self, topo_zones_specs):
        zones, specs = topo_zones_specs
        ids = [s["channel_id"] for s in specs]
        assert len(ids) == len(set(ids)), "Duplicate channel IDs found"

    def test_all_specs_have_required_fields(self, topo_zones_specs):
        zones, specs = topo_zones_specs
        for s in specs:
            assert "channel_id" in s
            assert "efuse_family" in s
            assert "zone_id" in s
            assert "load_name" in s


# ---------------------------------------------------------------------------
# Fleet-scale generation (65 channels)
# ---------------------------------------------------------------------------


class TestFleetScaleGeneration:
    def test_generate_65_channels(self):
        """Generate telemetry for all 65 example channels."""
        zones, specs = example_topology()
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
        assert unique_channels == 65
        # 5s / 100ms = 50 rows per channel × 65 channels = 3250
        assert len(telem_df) == 65 * 50

    def test_65ch_normalize_and_features(self):
        """Full pipeline on 65 channels: generate → normalize → features."""
        zones, specs = example_topology()
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
        assert feat_df["channel_id"].nunique() == 65

    def test_65ch_with_faults(self):
        """Generate with faults injected across multiple zones."""
        from src.schemas.telemetry import FaultInjection, FaultType

        zones, specs = example_topology()
        channels = build_channels(zones, specs)
        faults = [
            FaultInjection(
                channel_id="ch_017",
                fault_type=FaultType.OVERLOAD_SPIKE,
                start_s=1.0,
                duration_s=1.0,
                intensity=0.8,
            ),
            FaultInjection(
                channel_id="ch_041",
                fault_type=FaultType.VOLTAGE_SAG,
                start_s=2.0,
                duration_s=1.0,
                intensity=0.7,
            ),
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
    """Verify the example topology populates the IO-level fields."""

    def test_all_channels_have_system_cluster(self, topo_channels):
        for ch in topo_channels:
            assert ch.system_cluster != "", (
                f"{ch.channel_id} ({ch.load_name}) missing system_cluster"
            )

    def test_all_channels_have_system_name(self, topo_channels):
        for ch in topo_channels:
            assert ch.system_name != "", f"{ch.channel_id} ({ch.load_name}) missing system_name"

    def test_known_system_clusters(self, topo_channels):
        """All clusters should be from a known set."""
        clusters = {ch.system_cluster for ch in topo_channels}
        expected = {
            "adas",
            "auxiliary",
            "body_comfort",
            "drivetrain",
            "energy_management",
            "exterior_lighting",
            "infotainment",
            "interior_lighting",
            "occupant_safety",
            "power_distribution",
        }
        assert clusters == expected, f"Unexpected clusters: {clusters - expected}"

    def test_always_on_channels(self, topo_channels):
        """Safety-critical feeds should be always_on (KL30)."""
        always_on_names = {
            ch.load_name for ch in topo_channels if ch.power_class == PowerClass.ALWAYS_ON
        }
        assert "tail_light_left" in always_on_names
        assert "main_bus_safety" in always_on_names
        assert "power_supply_body_safety" in always_on_names

    def test_h_bridge_motors(self, topo_channels):
        """Bidirectional motors (door locks, closures) should use H-bridge."""
        locks = [ch for ch in topo_channels if "door_lock" in ch.load_name]
        for ch in locks:
            assert ch.driver_type == DriverType.H_BRIDGE, f"{ch.load_name} should be H-bridge"

    def test_h_bridge_closures(self, topo_channels):
        """Tailgate/deployable step should use H-bridge for bidirectional control."""
        hb = [ch for ch in topo_channels if ch.driver_type == DriverType.H_BRIDGE]
        assert any(ch.load_name == "tailgate_actuator" for ch in hb)
        assert any(ch.load_name == "deployable_step" for ch in hb)

    def test_pwm_capable_lights(self, topo_channels):
        """Dimmable lights should be PWM-capable."""
        reading = [ch for ch in topo_channels if ch.load_name == "interior_reading_light"][0]
        assert reading.pwm_capable is True
        stop = [ch for ch in topo_channels if ch.load_name == "center_stop_lamp"][0]
        assert stop.pwm_capable is True

    def test_non_pwm_defaults(self, topo_channels):
        """Simple on/off loads shouldn't be PWM-capable."""
        bus = [ch for ch in topo_channels if ch.load_name == "main_bus_safety"][0]
        assert bus.pwm_capable is False

    def test_capacitive_loads(self, topo_channels):
        """ECU/sensor power supplies should be capacitive load type."""
        charger = [ch for ch in topo_channels if ch.load_name == "wireless_charger_rear_1"][0]
        assert charger.load_type == "capacitive"
        radar = [ch for ch in topo_channels if ch.load_name == "corner_radar_rear_left"][0]
        assert radar.load_type == "capacitive"

    def test_default_driver_type_is_high_side(self, topo_channels):
        """Most channels should default to high-side driver."""
        hs_count = sum(1 for ch in topo_channels if ch.driver_type == DriverType.HIGH_SIDE)
        # Majority should be high-side (all except ~6 H-bridge)
        assert hs_count > 40

    def test_default_power_class_is_ignition(self, topo_channels):
        """Most channels should be ignition-switched (KL15)."""
        ign_count = sum(1 for ch in topo_channels if ch.power_class == PowerClass.IGNITION)
        assert ign_count > 40


# ---------------------------------------------------------------------------
# Schema additions
# ---------------------------------------------------------------------------


class TestSchemaAdditions:
    def test_efuse_family_enum_values(self):
        assert EFuseFamily.INF_HS_2A.value == "inf_hs_2a"
        assert EFuseFamily.ST_HS_50A.value == "st_hs_50a"
        assert EFuseFamily.CUSTOM.value == "custom"

    def test_zone_controller_defaults(self):
        zc = ZoneController(zone_id="test")
        assert zc.location == "body"
        assert zc.bus_interface == SourceProtocol.CAN
        assert zc.cdd_read_cycle_ms == 10.0

    def test_channel_meta_new_fields_backward_compatible(self):
        """ChannelMeta with no new fields should still work."""
        ch = ChannelMeta(channel_id="old_style", nominal_current_a=5.0, max_current_a=10.0)
        assert ch.efuse_family == EFuseFamily.INF_HS_14A
        assert ch.zone_id == ""
        assert ch.connected_loads == []

    def test_channel_meta_with_new_fields(self):
        ch = ChannelMeta(
            channel_id="new_style",
            efuse_family=EFuseFamily.ST_HS_30A,
            zone_id="zone_body",
            connected_loads=["seat_heater", "lumbar"],
            nominal_current_a=20.0,
            max_current_a=40.0,
        )
        assert ch.efuse_family == EFuseFamily.ST_HS_30A
        assert ch.zone_id == "zone_body"
        assert len(ch.connected_loads) == 2
