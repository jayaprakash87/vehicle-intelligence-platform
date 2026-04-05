"""eFuse catalog and vehicle topology factory.

Instead of hand-writing 50+ channel definitions, users define:
  1. Zone Controllers (physical ECUs hosting eFuse ICs)
  2. Channel assignments referencing eFuse families from the catalog
  3. The factory expands this topology into fully specified ChannelMeta list

The catalog provides electrical / thermal defaults per eFuse IC family.
Individual channels can override any parameter.

Architecture reference:
    eFuse IC (HW) → SPI → CDD (Complex Device Driver, SW) → COM → CAN/LIN
    Zone Controller = physical ECU running the CDD software

Supported IC families:
    Infineon PROFET+2 (BTS70xx), TLE92104, BTS81000
    ST VIPower VN (single HS), VND (dual HS), VNH (H-bridge), VNL (low-side)
    CUSTOM — user-defined ASIC with user-provided parameters
"""

from __future__ import annotations

from src.schemas.telemetry import (
    ChannelMeta,
    EFuseFamily,
    EFuseProfile,
    SafetyLevel,
    ZoneController,
)

# ---------------------------------------------------------------------------
# Catalog of eFuse IC families with realistic electrical parameters
# ---------------------------------------------------------------------------
# Each entry maps an EFuseFamily to a specific production IC.
#
# Sources: Infineon PROFET+2 datasheets (BTS70xx), Infineon TLE92104 / BTS81000,
# STMicroelectronics VIPower datasheets (VN, VND, VNH, VNL).
#
# Key relationships:
#   - r_ds_on falls with higher current ratings (bigger die / parallel FETs)
#   - r_thermal_kw in °C/W (junction-to-ambient, with PCB copper)
#   - tau_thermal_s = R_th × C_th time constant
#   - Multi-channel ICs (e.g. TLE92104 4ch) share thermal mass on the die
#
# CUSTOM entry provides safe defaults — users MUST override for real ASICs.

EFUSE_CATALOG: dict[EFuseFamily, EFuseProfile] = {
    # ── Infineon PROFET+2 — BTS70xx, Rdson-parametric ────────────────────────
    #
    # BTS7040-1EPA: 40mΩ, ~2.8A, single-channel
    EFuseFamily.INF_HS_2A: EFuseProfile(
        efuse_family=EFuseFamily.INF_HS_2A,
        ic_part_number="BTS7040-1EPA",
        manufacturer="Infineon",
        nominal_current_a=1.5,
        max_current_a=4.0,
        fuse_rating_a=2.8,
        r_ds_on_ohm=0.040,
        r_thermal_kw=80.0,
        tau_thermal_s=8.0,
        cooldown_s=0.5,
        max_retries=5,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=8500.0,  # BTS7040 ILIS ratio (datasheet typ.)
    ),
    # BTS7020-2EPA: 20mΩ, ~5.5A, dual-channel
    EFuseFamily.INF_HS_5A: EFuseProfile(
        efuse_family=EFuseFamily.INF_HS_5A,
        ic_part_number="BTS7020-2EPA",
        manufacturer="Infineon",
        nominal_current_a=3.5,
        max_current_a=8.0,
        fuse_rating_a=5.5,
        r_ds_on_ohm=0.020,
        r_thermal_kw=65.0,
        tau_thermal_s=10.0,
        cooldown_s=0.5,
        max_retries=5,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=6400.0,  # BTS7020 ILIS ratio
    ),
    # BTS7012-1EPA: 12mΩ, ~9A, single-channel
    EFuseFamily.INF_HS_9A: EFuseProfile(
        efuse_family=EFuseFamily.INF_HS_9A,
        ic_part_number="BTS7012-1EPA",
        manufacturer="Infineon",
        nominal_current_a=6.0,
        max_current_a=12.0,
        fuse_rating_a=9.0,
        r_ds_on_ohm=0.012,
        r_thermal_kw=50.0,
        tau_thermal_s=12.0,
        cooldown_s=0.8,
        max_retries=4,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=4580.0,  # BTS7012 ILIS ratio
    ),
    # BTS7010-1EPA: 10mΩ, ~11A, single-channel
    EFuseFamily.INF_HS_11A: EFuseProfile(
        efuse_family=EFuseFamily.INF_HS_11A,
        ic_part_number="BTS7010-1EPA",
        manufacturer="Infineon",
        nominal_current_a=7.0,
        max_current_a=15.0,
        fuse_rating_a=11.0,
        r_ds_on_ohm=0.010,
        r_thermal_kw=45.0,
        tau_thermal_s=14.0,
        cooldown_s=1.0,
        max_retries=4,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=3640.0,  # BTS7010 ILIS ratio
    ),
    # BTS7008-1EPA: 8mΩ, ~14A, single-channel
    EFuseFamily.INF_HS_14A: EFuseProfile(
        efuse_family=EFuseFamily.INF_HS_14A,
        ic_part_number="BTS7008-1EPA",
        manufacturer="Infineon",
        nominal_current_a=9.0,
        max_current_a=18.0,
        fuse_rating_a=14.0,
        r_ds_on_ohm=0.008,
        r_thermal_kw=40.0,
        tau_thermal_s=15.0,
        cooldown_s=1.0,
        max_retries=3,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=3640.0,  # BTS7008 ILIS ratio
    ),
    # BTS7006-1EPZ: 6mΩ, ~18A, single-channel
    EFuseFamily.INF_HS_18A: EFuseProfile(
        efuse_family=EFuseFamily.INF_HS_18A,
        ic_part_number="BTS7006-1EPZ",
        manufacturer="Infineon",
        nominal_current_a=12.0,
        max_current_a=24.0,
        fuse_rating_a=18.0,
        r_ds_on_ohm=0.006,
        r_thermal_kw=35.0,
        tau_thermal_s=18.0,
        cooldown_s=1.0,
        max_retries=3,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=2830.0,  # BTS7006 ILIS ratio
    ),
    # BTS7004-1EPP: 4mΩ, ~28A, single-channel
    EFuseFamily.INF_HS_28A: EFuseProfile(
        efuse_family=EFuseFamily.INF_HS_28A,
        ic_part_number="BTS7004-1EPP",
        manufacturer="Infineon",
        nominal_current_a=18.0,
        max_current_a=35.0,
        fuse_rating_a=28.0,
        r_ds_on_ohm=0.004,
        r_thermal_kw=28.0,
        tau_thermal_s=20.0,
        cooldown_s=1.5,
        max_retries=3,
        current_adc_bits=14,
        load_type="resistive",
        k_ilis=1550.0,  # BTS7004 ILIS ratio (large die → lower ratio)
    ),
    # ── Infineon multi-channel and high-current ──────────────────────────────
    #
    # TLE92104-232QX: 4ch smart switch, ≤10A/ch (only HS used per R30 spec)
    EFuseFamily.INF_MULTI_10A: EFuseProfile(
        efuse_family=EFuseFamily.INF_MULTI_10A,
        ic_part_number="TLE92104-232QX",
        manufacturer="Infineon",
        nominal_current_a=6.0,
        max_current_a=15.0,
        fuse_rating_a=10.0,
        r_ds_on_ohm=0.025,
        r_thermal_kw=40.0,
        tau_thermal_s=15.0,
        cooldown_s=1.0,
        max_retries=3,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=3640.0,  # TLE92104 ILIS ratio
        safety_level=SafetyLevel.ASIL_B,
    ),
    # BTS81000-SSGI-6ET: high-current PDU, ≤100A
    EFuseFamily.INF_HS_100A: EFuseProfile(
        efuse_family=EFuseFamily.INF_HS_100A,
        ic_part_number="BTS81000-SSGI-6ET",
        manufacturer="Infineon",
        nominal_current_a=60.0,
        max_current_a=120.0,
        fuse_rating_a=100.0,
        r_ds_on_ohm=0.001,
        r_thermal_kw=15.0,
        tau_thermal_s=35.0,
        cooldown_s=3.0,
        max_retries=2,
        current_adc_bits=10,
        load_type="resistive",
        k_ilis=1000.0,  # BTS81000 high-current sense ratio
        safety_level=SafetyLevel.ASIL_B,
    ),
    # ── ST VIPower single high-side ──────────────────────────────────────────
    #
    # VN7140AS: single HS, ~14A
    EFuseFamily.ST_HS_14A: EFuseProfile(
        efuse_family=EFuseFamily.ST_HS_14A,
        ic_part_number="VN7140AS",
        manufacturer="STMicroelectronics",
        nominal_current_a=9.0,
        max_current_a=18.0,
        fuse_rating_a=14.0,
        r_ds_on_ohm=0.040,
        r_thermal_kw=42.0,
        tau_thermal_s=14.0,
        cooldown_s=1.0,
        max_retries=4,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=7500.0,  # ST VN7140 CS ratio
    ),
    # VN9E30F: single HS, ~30A
    EFuseFamily.ST_HS_30A: EFuseProfile(
        efuse_family=EFuseFamily.ST_HS_30A,
        ic_part_number="VN9E30F",
        manufacturer="STMicroelectronics",
        nominal_current_a=20.0,
        max_current_a=40.0,
        fuse_rating_a=30.0,
        r_ds_on_ohm=0.005,
        r_thermal_kw=25.0,
        tau_thermal_s=22.0,
        cooldown_s=2.0,
        max_retries=2,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=4000.0,  # ST VN9E30F CS ratio
    ),
    # VN7050AS: single HS, ~50A
    EFuseFamily.ST_HS_50A: EFuseProfile(
        efuse_family=EFuseFamily.ST_HS_50A,
        ic_part_number="VN7050AS",
        manufacturer="STMicroelectronics",
        nominal_current_a=35.0,
        max_current_a=65.0,
        fuse_rating_a=50.0,
        r_ds_on_ohm=0.003,
        r_thermal_kw=18.0,
        tau_thermal_s=28.0,
        cooldown_s=2.5,
        max_retries=2,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=2500.0,  # ST VN7050 CS ratio
    ),
    # ── ST VIPower dual, H-bridge, low-side ──────────────────────────────────
    #
    # VND7140AJ: dual HS, ~14A/ch
    EFuseFamily.ST_DUAL_14A: EFuseProfile(
        efuse_family=EFuseFamily.ST_DUAL_14A,
        ic_part_number="VND7140AJ",
        manufacturer="STMicroelectronics",
        nominal_current_a=9.0,
        max_current_a=18.0,
        fuse_rating_a=14.0,
        r_ds_on_ohm=0.060,
        r_thermal_kw=38.0,
        tau_thermal_s=16.0,
        cooldown_s=1.0,
        max_retries=3,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=7500.0,  # ST VND7140 CS ratio
    ),
    # VNH9045AQTR: H-bridge motor driver, ~30A
    EFuseFamily.ST_HB_30A: EFuseProfile(
        efuse_family=EFuseFamily.ST_HB_30A,
        ic_part_number="VNH9045AQTR",
        manufacturer="STMicroelectronics",
        nominal_current_a=20.0,
        max_current_a=40.0,
        fuse_rating_a=30.0,
        r_ds_on_ohm=0.015,
        r_thermal_kw=30.0,
        tau_thermal_s=18.0,
        cooldown_s=1.5,
        max_retries=3,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=4000.0,  # ST VNH9045 CS ratio
    ),
    # VNL5050S5-E: low-side, ~50A
    EFuseFamily.ST_LS_50A: EFuseProfile(
        efuse_family=EFuseFamily.ST_LS_50A,
        ic_part_number="VNL5050S5-E",
        manufacturer="STMicroelectronics",
        nominal_current_a=35.0,
        max_current_a=65.0,
        fuse_rating_a=50.0,
        r_ds_on_ohm=0.004,
        r_thermal_kw=20.0,
        tau_thermal_s=25.0,
        cooldown_s=2.0,
        max_retries=2,
        current_adc_bits=10,
        load_type="resistive",
        k_ilis=2500.0,  # ST VNL5050 CS ratio
    ),
    # ── Custom / ASIC ─────────────────────────────────────────────────────────
    # Safe generic defaults — users MUST override for real custom ASICs.
    EFuseFamily.CUSTOM: EFuseProfile(
        efuse_family=EFuseFamily.CUSTOM,
        ic_part_number="CUSTOM_ASIC",
        manufacturer="custom",
        nominal_current_a=5.0,
        max_current_a=10.0,
        fuse_rating_a=8.0,
        r_ds_on_ohm=0.025,
        r_thermal_kw=50.0,
        tau_thermal_s=12.0,
        cooldown_s=1.0,
        max_retries=3,
        current_adc_bits=12,
        load_type="resistive",
        k_ilis=5000.0,  # CUSTOM — override for real ASIC
    ),
}


def get_profile(family: EFuseFamily) -> EFuseProfile:
    """Look up the default electrical profile for an eFuse family."""
    return EFUSE_CATALOG[family]


# ---------------------------------------------------------------------------
# Vehicle topology definition — compact input format
# ---------------------------------------------------------------------------


class ChannelSpec(dict):
    """Lightweight dict subclass for type clarity in YAML configs.

    Keys map directly to ChannelMeta fields. At minimum:
        channel_id, load_name, efuse_family
    Optional overrides: nominal_current_a, load_type, inrush_factor, etc.
    Any field not specified is inherited from the eFuse catalog.
    """

    pass


# ---------------------------------------------------------------------------
# Channel factory — expand topology into full ChannelMeta list
# ---------------------------------------------------------------------------


def build_channels(
    zones: list[ZoneController],
    channel_specs: list[dict],
) -> list[ChannelMeta]:
    """Expand compact channel specs into fully specified ChannelMeta objects.

    For each spec:
      1. Look up the eFuse family profile from the catalog
      2. Apply catalog defaults for all electrical / thermal params
      3. Apply any per-channel overrides from the spec
      4. Inherit zone_id and source_protocol from the assigned Zone Controller

    Parameters
    ----------
    zones : list[ZoneController]
        Zone Controllers in this vehicle.
    channel_specs : list[dict]
        Minimal channel definitions. Required keys: channel_id, efuse_family.
        Optional: load_name, zone_id, connected_loads, load_type, and any
        ChannelMeta field to override catalog defaults.

    Returns
    -------
    list[ChannelMeta]
        Fully populated channel list ready for SimulationConfig.
    """
    zone_map = {z.zone_id: z for z in zones}
    channels: list[ChannelMeta] = []

    for spec in channel_specs:
        spec = dict(spec)  # shallow copy

        # Resolve eFuse family
        family_raw = spec.pop("efuse_family", "hs_15a")
        if isinstance(family_raw, str):
            family = EFuseFamily(family_raw)
        else:
            family = family_raw
        profile = get_profile(family)

        # Start from catalog defaults
        defaults = {
            "efuse_family": family,
            "nominal_current_a": profile.nominal_current_a,
            "max_current_a": profile.max_current_a,
            "fuse_rating_a": profile.fuse_rating_a,
            "r_ds_on_ohm": profile.r_ds_on_ohm,
            "r_thermal_kw": profile.r_thermal_kw,
            "tau_thermal_s": profile.tau_thermal_s,
            "cooldown_s": profile.cooldown_s,
            "max_retries": profile.max_retries,
            "current_adc_bits": profile.current_adc_bits,
            "voltage_adc_bits": profile.voltage_adc_bits,
            "fit_threshold_a2s": profile.fit_threshold_a2s,
            "short_circuit_threshold_a": profile.short_circuit_threshold_a,
            "thermal_shutdown_c": profile.thermal_shutdown_c,
            "load_type": profile.load_type,
            "k_ilis": profile.k_ilis,
            "k_ilis_tempco_ppm_c": profile.k_ilis_tempco_ppm_c,
            "r_ilis_ohm": profile.r_ilis_ohm,
            "r_ilis_tolerance": profile.r_ilis_tolerance,
            "ol_blank_time_ms": profile.ol_blank_time_ms,
            "ol_threshold_a": profile.ol_threshold_a,
        }

        # Inherit from zone controller if assigned
        zone_id = spec.get("zone_id", "")
        if zone_id and zone_id in zone_map:
            zone = zone_map[zone_id]
            defaults["source_protocol"] = zone.bus_interface

        # Merge: catalog defaults < spec overrides
        merged = {**defaults, **spec}
        channels.append(ChannelMeta(**merged))

    return channels


# ---------------------------------------------------------------------------
# Example vehicle topology — canonical 65-channel reference
# ---------------------------------------------------------------------------


def example_topology() -> tuple[list[ZoneController], list[dict]]:
    """Return a generic zonal EE topology with ~65 eFuse channels.

    Demonstrates a modern 4-zone controller architecture typical of BEV /
    premium platforms.  Zone names and load assignments are intentionally
    generic so the topology serves as a starting template — users should
    adapt zone composition, channel counts, and system clusters to match
    their own vehicle program.

    Zones:
        zone_rear     (25 ch) — seating, lighting, infotainment, ADAS, drivetrain
        zone_body     (15 ch) — doors, locks, cabin climate, body electronics
        zone_front    (15 ch) — power supply, HVAC, suspension, auxiliary loads
        zone_central  (10 ch) — power distribution, closures, reserves
    """
    zones = [
        ZoneController(zone_id="zone_rear", name="Rear Zone Controller", location="rear"),
        ZoneController(zone_id="zone_body", name="Body Zone Controller", location="body"),
        ZoneController(zone_id="zone_front", name="Front Zone Controller", location="front"),
        ZoneController(
            zone_id="zone_central", name="Central Zone Controller", location="underhood"
        ),
    ]

    specs: list[dict] = []
    _ch = 0

    def _add(
        zone_id: str,
        family: str,
        load_name: str,
        load_type: str = "resistive",
        connected_loads: list[str] | None = None,
        system_cluster: str = "",
        system_name: str = "",
        driver_type: str = "high_side",
        power_class: str = "ignition",
        pwm_capable: bool = False,
        **kw,
    ) -> None:
        nonlocal _ch
        _ch += 1
        s: dict = {
            "channel_id": f"ch_{_ch:03d}",
            "zone_id": zone_id,
            "efuse_family": family,
            "load_name": load_name,
            "load_type": load_type,
            "connected_loads": connected_loads or [load_name],
            "system_cluster": system_cluster,
            "system_name": system_name,
            "driver_type": driver_type,
            "power_class": power_class,
            "pwm_capable": pwm_capable,
        }
        s.update(kw)
        specs.append(s)

    # ── Rear zone (25 channels) ──────────────────────────────────
    # Seating comfort
    _add("zone_rear", "inf_hs_11a", "seat_adjust_rear",
         system_cluster="body_comfort", system_name="seat_adjustment")
    _add("zone_rear", "inf_hs_9a", "seat_heater_rear_right",
         load_type="ptc", pwm_capable=True,
         system_cluster="body_comfort", system_name="seat_heating")
    _add("zone_rear", "inf_hs_9a", "seat_heater_rear_left",
         load_type="ptc", pwm_capable=True,
         system_cluster="body_comfort", system_name="seat_heating")
    _add("zone_rear", "inf_hs_5a", "seat_ventilation_pump",
         load_type="motor", inrush_factor=3.0, inrush_duration_ms=40.0,
         system_cluster="body_comfort", system_name="seat_ventilation")
    _add("zone_rear", "inf_hs_5a", "seatbelt_heater_rear",
         load_type="ptc", pwm_capable=True,
         system_cluster="body_comfort", system_name="heated_surfaces")
    # Infotainment
    _add("zone_rear", "inf_hs_2a", "wireless_charger_rear_1",
         load_type="capacitive",
         system_cluster="infotainment", system_name="connectivity")
    _add("zone_rear", "inf_hs_2a", "wireless_charger_rear_2",
         load_type="capacitive",
         system_cluster="infotainment", system_name="connectivity")
    _add("zone_rear", "inf_hs_5a", "media_hub_rear",
         load_type="capacitive",
         system_cluster="infotainment", system_name="media_interface")
    # Exterior lighting
    _add("zone_rear", "inf_hs_9a", "tail_light_left",
         connected_loads=["tail_light_left", "brake_light_left"],
         system_cluster="exterior_lighting", system_name="rear_lighting",
         pwm_capable=True, power_class="always_on")
    _add("zone_rear", "inf_hs_9a", "tail_light_right",
         connected_loads=["tail_light_right", "brake_light_right"],
         system_cluster="exterior_lighting", system_name="rear_lighting",
         pwm_capable=True, power_class="always_on")
    _add("zone_rear", "inf_hs_5a", "center_stop_lamp",
         system_cluster="exterior_lighting", system_name="rear_lighting",
         pwm_capable=True)
    _add("zone_rear", "inf_hs_11a", "rear_defroster",
         load_type="ptc", pwm_capable=True,
         system_cluster="body_comfort", system_name="climate_support")
    # Audio
    _add("zone_rear", "inf_hs_14a", "amplifier_main",
         system_cluster="infotainment", system_name="audio_system")
    _add("zone_rear", "inf_hs_14a", "amplifier_subwoofer",
         system_cluster="infotainment", system_name="audio_system")
    # Energy / charging
    _add("zone_rear", "inf_hs_18a", "charge_port_controller",
         system_cluster="energy_management", system_name="charging")
    _add("zone_rear", "inf_hs_5a", "power_outlet_rear",
         system_cluster="energy_management", system_name="auxiliary_power")
    _add("zone_rear", "inf_hs_5a", "socket_12v_rear",
         system_cluster="energy_management", system_name="auxiliary_power")
    # Drivetrain / chassis
    _add("zone_rear", "inf_hs_28a", "rear_drive_inverter",
         system_cluster="drivetrain", system_name="electric_drive")
    _add("zone_rear", "st_hb_30a", "deployable_step",
         load_type="motor", inrush_factor=5.0, inrush_duration_ms=80.0,
         system_cluster="body_comfort", system_name="convenience_actuators",
         driver_type="h_bridge")
    _add("zone_rear", "inf_hs_14a", "rear_steer_actuator",
         system_cluster="drivetrain", system_name="rear_axle_steering")
    # ADAS
    _add("zone_rear", "inf_hs_2a", "corner_radar_rear_left",
         load_type="capacitive",
         system_cluster="adas", system_name="surround_sensing")
    _add("zone_rear", "inf_hs_2a", "corner_radar_rear_right",
         load_type="capacitive",
         system_cluster="adas", system_name="surround_sensing")
    # Occupant safety
    _add("zone_rear", "inf_hs_5a", "belt_pretensioner_left",
         system_cluster="occupant_safety", system_name="restraint_systems")
    _add("zone_rear", "inf_hs_5a", "belt_pretensioner_right",
         system_cluster="occupant_safety", system_name="restraint_systems")
    # Suspension
    _add("zone_rear", "inf_hs_18a", "active_damper_rear",
         system_cluster="drivetrain", system_name="active_suspension")

    # ── Body zone (15 channels) ──────────────────────────────────
    # Door modules
    _add("zone_body", "inf_hs_11a", "door_lock_front_left",
         load_type="motor", inrush_factor=4.0, inrush_duration_ms=40.0,
         system_cluster="body_comfort", system_name="door_modules",
         driver_type="h_bridge")
    _add("zone_body", "inf_hs_11a", "door_lock_front_right",
         load_type="motor", inrush_factor=4.0, inrush_duration_ms=40.0,
         system_cluster="body_comfort", system_name="door_modules",
         driver_type="h_bridge")
    _add("zone_body", "inf_hs_11a", "door_lock_rear_left",
         load_type="motor", inrush_factor=4.0, inrush_duration_ms=40.0,
         system_cluster="body_comfort", system_name="door_modules",
         driver_type="h_bridge")
    _add("zone_body", "inf_hs_11a", "door_lock_rear_right",
         load_type="motor", inrush_factor=4.0, inrush_duration_ms=40.0,
         system_cluster="body_comfort", system_name="door_modules",
         driver_type="h_bridge")
    _add("zone_body", "inf_hs_28a", "pdu_main_feed",
         system_cluster="power_distribution", system_name="body_power",
         safety_level="asil_b")
    _add("zone_body", "inf_hs_14a", "keyless_entry_module",
         system_cluster="body_comfort", system_name="keyless_access")
    _add("zone_body", "inf_hs_14a", "immobilizer_relay",
         system_cluster="body_comfort", system_name="keyless_access")
    # Infrastructure power
    _add("zone_body", "inf_hs_100a", "power_supply_body_safety",
         system_cluster="power_distribution", system_name="infrastructure_power",
         safety_level="asil_b", power_class="always_on")
    _add("zone_body", "inf_hs_100a", "power_supply_body_comfort",
         system_cluster="power_distribution", system_name="infrastructure_power",
         power_class="always_on")
    # Cabin climate
    _add("zone_body", "inf_hs_9a", "hvac_blend_door",
         system_cluster="body_comfort", system_name="cabin_climate")
    _add("zone_body", "inf_hs_18a", "ptc_cabin_heater",
         load_type="ptc", pwm_capable=True,
         system_cluster="body_comfort", system_name="cabin_climate")
    _add("zone_body", "inf_hs_2a", "cabin_air_quality_sensor",
         system_cluster="body_comfort", system_name="cabin_climate")
    # Body electronics
    _add("zone_body", "inf_hs_5a", "steering_column_heater",
         load_type="ptc", pwm_capable=True,
         system_cluster="body_comfort", system_name="heated_surfaces")
    _add("zone_body", "inf_hs_5a", "rear_climate_panel",
         system_cluster="body_comfort", system_name="cabin_climate")
    _add("zone_body", "inf_hs_2a", "cooled_storage_compartment",
         system_cluster="body_comfort", system_name="convenience_features")

    # ── Front zone (15 channels) ─────────────────────────────────
    # Infrastructure power
    _add("zone_front", "inf_hs_100a", "power_supply_front_safety",
         system_cluster="power_distribution", system_name="infrastructure_power",
         safety_level="asil_b", power_class="always_on")
    _add("zone_front", "inf_hs_100a", "power_supply_front_aux",
         system_cluster="power_distribution", system_name="infrastructure_power",
         power_class="always_on")
    # HVAC
    _add("zone_front", "inf_hs_18a", "coolant_heater",
         load_type="ptc", pwm_capable=True,
         system_cluster="body_comfort", system_name="cabin_climate")
    _add("zone_front", "st_hs_50a", "ac_compressor",
         load_type="motor", inrush_factor=5.0, inrush_duration_ms=80.0,
         system_cluster="body_comfort", system_name="cabin_climate",
         pwm_capable=True)
    _add("zone_front", "inf_hs_14a", "blower_motor",
         load_type="motor", inrush_factor=4.0, inrush_duration_ms=60.0,
         system_cluster="body_comfort", system_name="cabin_climate",
         pwm_capable=True)
    # Suspension / chassis
    _add("zone_front", "inf_hs_18a", "active_damper_front",
         system_cluster="drivetrain", system_name="active_suspension")
    _add("zone_front", "inf_hs_14a", "supercap_suspension",
         load_type="capacitive",
         system_cluster="drivetrain", system_name="active_suspension")
    # Seat comfort (front)
    _add("zone_front", "st_dual_14a", "massage_seat_left",
         system_cluster="body_comfort", system_name="seat_comfort")
    _add("zone_front", "st_dual_14a", "massage_seat_right",
         system_cluster="body_comfort", system_name="seat_comfort")
    # Auxiliary loads / reserves
    _add("zone_front", "inf_hs_5a", "aux_load_1",
         system_cluster="auxiliary", system_name="auxiliary_loads")
    _add("zone_front", "inf_hs_5a", "aux_load_2",
         system_cluster="auxiliary", system_name="auxiliary_loads")
    _add("zone_front", "inf_hs_5a", "aux_load_3",
         system_cluster="auxiliary", system_name="auxiliary_loads")
    # Body / access
    _add("zone_front", "inf_hs_14a", "pdu_cross_feed",
         system_cluster="power_distribution", system_name="body_power")
    _add("zone_front", "inf_hs_5a", "tire_pressure_module",
         load_type="capacitive",
         system_cluster="adas", system_name="tire_monitoring")
    _add("zone_front", "inf_hs_2a", "reserve_channel",
         system_cluster="auxiliary", system_name="auxiliary_loads")

    # ── Central zone (10 channels) ───────────────────────────────
    # Sensors / misc
    _add("zone_central", "inf_hs_2a", "climate_sensor_module",
         load_type="capacitive",
         system_cluster="body_comfort", system_name="sensor_modules")
    _add("zone_central", "st_hb_30a", "tailgate_actuator",
         load_type="motor", inrush_factor=5.0, inrush_duration_ms=60.0,
         system_cluster="body_comfort", system_name="closure_actuators",
         driver_type="h_bridge")
    _add("zone_central", "inf_hs_2a", "interior_reading_light",
         pwm_capable=True,
         system_cluster="interior_lighting", system_name="cabin_lights")
    # Power distribution
    _add("zone_central", "inf_hs_100a", "main_bus_safety",
         system_cluster="power_distribution", system_name="main_bus",
         safety_level="asil_b", power_class="always_on")
    _add("zone_central", "inf_hs_100a", "main_bus_aux",
         system_cluster="power_distribution", system_name="main_bus",
         power_class="always_on")
    # Infrastructure feeds
    _add("zone_central", "inf_hs_100a", "power_supply_central_safety",
         system_cluster="power_distribution", system_name="infrastructure_power",
         safety_level="asil_b", power_class="always_on")
    _add("zone_central", "inf_hs_100a", "power_supply_central_aux",
         system_cluster="power_distribution", system_name="infrastructure_power",
         power_class="always_on")
    # Reserves
    _add("zone_central", "inf_hs_5a", "reserve_1",
         system_cluster="auxiliary", system_name="auxiliary_loads")
    _add("zone_central", "inf_hs_5a", "reserve_2",
         system_cluster="auxiliary", system_name="auxiliary_loads")
    _add("zone_central", "inf_hs_5a", "reserve_3",
         system_cluster="auxiliary", system_name="auxiliary_loads")

    assert _ch == 65, f"Expected 65 channels, got {_ch}"
    return zones, specs
