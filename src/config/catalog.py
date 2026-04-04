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
# Each entry maps an EFuseFamily (current class + topology) to a specific
# representative IC.  Real vehicle programs select ICs per output; our
# catalog picks one credible default per class.
#
# Sources: Infineon PROFET+ / PROFET+2 datasheets, STMicroelectronics
# VIPower (VNx, VNDx) datasheets, typical OEM integration specs.
#
# Key relationships:
#   - r_ds_on falls with higher current ratings (bigger die / parallel FETs)
#   - r_thermal_kw in °C/W (junction-to-ambient, with PCB copper)
#   - tau_thermal_s = R_th × C_th time constant
#   - Multi-channel ICs (e.g. TLE92104 4ch) share thermal mass on the die

EFUSE_CATALOG: dict[EFuseFamily, EFuseProfile] = {
    # --- BTS7006-1EPP: Infineon PROFET+2, 1ch high-side, ~2.5A class -------
    EFuseFamily.HS_2A: EFuseProfile(
        efuse_family=EFuseFamily.HS_2A,
        ic_part_number="BTS7006-1EPP",
        manufacturer="Infineon",
        nominal_current_a=1.5,
        max_current_a=3.0,
        fuse_rating_a=2.5,
        r_ds_on_ohm=0.180,
        r_thermal_kw=80.0,
        tau_thermal_s=8.0,
        cooldown_s=0.5,
        max_retries=5,
        current_adc_bits=12,
        load_type="resistive",
    ),
    # --- VN7140AS: STMicro VIPower, 1ch high-side, ~5A class ----------------
    EFuseFamily.HS_5A: EFuseProfile(
        efuse_family=EFuseFamily.HS_5A,
        ic_part_number="VN7140AS",
        manufacturer="STMicroelectronics",
        nominal_current_a=3.5,
        max_current_a=7.0,
        fuse_rating_a=5.0,
        r_ds_on_ohm=0.060,
        r_thermal_kw=55.0,
        tau_thermal_s=12.0,
        cooldown_s=0.5,
        max_retries=4,
        current_adc_bits=12,
        load_type="resistive",
    ),
    # --- TLE92104: Infineon, 4ch low-side/high-side smart switch, ~10A class -
    EFuseFamily.HS_10A: EFuseProfile(
        efuse_family=EFuseFamily.HS_10A,
        ic_part_number="TLE92104",
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
        safety_level=SafetyLevel.ASIL_B,
    ),
    # --- VND7140AJ: STMicro, dual high-side driver, ~15A class ---------------
    EFuseFamily.HS_15A: EFuseProfile(
        efuse_family=EFuseFamily.HS_15A,
        ic_part_number="VND7140AJ",
        manufacturer="STMicroelectronics",
        nominal_current_a=10.0,
        max_current_a=20.0,
        fuse_rating_a=15.0,
        r_ds_on_ohm=0.012,
        r_thermal_kw=35.0,
        tau_thermal_s=18.0,
        cooldown_s=1.0,
        max_retries=3,
        current_adc_bits=12,
        load_type="resistive",
    ),
    # --- BTS7004-1EPP: Infineon PROFET+2, 1ch high-side, ~20A class ----------
    EFuseFamily.HS_20A: EFuseProfile(
        efuse_family=EFuseFamily.HS_20A,
        ic_part_number="BTS7004-1EPP",
        manufacturer="Infineon",
        nominal_current_a=14.0,
        max_current_a=28.0,
        fuse_rating_a=20.0,
        r_ds_on_ohm=0.008,
        r_thermal_kw=30.0,
        tau_thermal_s=20.0,
        cooldown_s=1.5,
        max_retries=3,
        current_adc_bits=14,
        load_type="resistive",
    ),
    # --- VN9×E30F: STMicro VIPower, 1ch high-side, ~30A class ----------------
    EFuseFamily.HS_30A: EFuseProfile(
        efuse_family=EFuseFamily.HS_30A,
        ic_part_number="VN9xE30F",
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
    ),
    # --- BTS81000-SSGI: Infineon, 1ch high-side, ~50A class ------------------
    EFuseFamily.HS_50A: EFuseProfile(
        efuse_family=EFuseFamily.HS_50A,
        ic_part_number="BTS81000-SSGI",
        manufacturer="Infineon",
        nominal_current_a=35.0,
        max_current_a=65.0,
        fuse_rating_a=50.0,
        r_ds_on_ohm=0.003,
        r_thermal_kw=18.0,
        tau_thermal_s=30.0,
        cooldown_s=3.0,
        max_retries=2,
        current_adc_bits=10,
        load_type="resistive",
        safety_level=SafetyLevel.ASIL_B,
    ),
    # --- TLE92104 (LS config): Infineon, 4ch low-side, ~5A class -------------
    EFuseFamily.LS_5A: EFuseProfile(
        efuse_family=EFuseFamily.LS_5A,
        ic_part_number="TLE92104",
        manufacturer="Infineon",
        nominal_current_a=3.0,
        max_current_a=7.0,
        fuse_rating_a=5.0,
        r_ds_on_ohm=0.070,
        r_thermal_kw=60.0,
        tau_thermal_s=10.0,
        cooldown_s=0.5,
        max_retries=4,
        current_adc_bits=12,
        load_type="resistive",
    ),
    # --- VNH9045: STMicro, H-bridge motor driver, ~15A class ------------------
    EFuseFamily.LS_15A: EFuseProfile(
        efuse_family=EFuseFamily.LS_15A,
        ic_part_number="VNH9045",
        manufacturer="STMicroelectronics",
        nominal_current_a=10.0,
        max_current_a=20.0,
        fuse_rating_a=15.0,
        r_ds_on_ohm=0.015,
        r_thermal_kw=35.0,
        tau_thermal_s=15.0,
        cooldown_s=1.0,
        max_retries=3,
        current_adc_bits=12,
        load_type="resistive",
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
# Realistic vehicle topology templates
# ---------------------------------------------------------------------------


def sedan_topology() -> tuple[list[ZoneController], list[dict]]:
    """Return a representative mid-size sedan topology with 52 eFuse channels.

    Zones:
        body (16 ch)  — interior lights, mirrors, locks, windows
        front (14 ch) — headlamps, wipers, horn, washer, ADAS sensors
        rear (10 ch)  — tail lights, defroster, trunk, parking sensors
        underhood (12 ch) — engine fan, fuel pump, starter relay, HVAC
    """
    zones = [
        ZoneController(zone_id="zone_body", name="Body Zone Controller", location="body"),
        ZoneController(zone_id="zone_front", name="Front Zone Controller", location="front"),
        ZoneController(zone_id="zone_rear", name="Rear Zone Controller", location="rear"),
        ZoneController(
            zone_id="zone_underhood", name="Underhood Zone Controller", location="underhood"
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

    # ── Body zone (16 channels) ──────────────────────────────────
    _add(
        "zone_body",
        "hs_2a",
        "dome_light",
        connected_loads=["dome_light", "map_light_left", "map_light_right"],
        system_cluster="interior_lighting",
        system_name="cabin_lights",
        pwm_capable=True,
    )
    _add(
        "zone_body",
        "hs_2a",
        "ambient_led_driver",
        connected_loads=["ambient_footwell", "ambient_door"],
        system_cluster="interior_lighting",
        system_name="ambient_lights",
        pwm_capable=True,
    )
    _add(
        "zone_body",
        "hs_2a",
        "courtesy_light_left",
        system_cluster="interior_lighting",
        system_name="entry_lights",
        pwm_capable=True,
    )
    _add(
        "zone_body",
        "hs_2a",
        "courtesy_light_right",
        system_cluster="interior_lighting",
        system_name="entry_lights",
        pwm_capable=True,
    )
    _add(
        "zone_body",
        "hs_5a",
        "mirror_fold_left",
        load_type="motor",
        inrush_factor=3.0,
        inrush_duration_ms=30.0,
        system_cluster="body_comfort",
        system_name="mirrors",
        driver_type="h_bridge",
    )
    _add(
        "zone_body",
        "hs_5a",
        "mirror_fold_right",
        load_type="motor",
        inrush_factor=3.0,
        inrush_duration_ms=30.0,
        system_cluster="body_comfort",
        system_name="mirrors",
        driver_type="h_bridge",
    )
    _add(
        "zone_body",
        "hs_5a",
        "mirror_heater_left",
        load_type="ptc",
        system_cluster="body_comfort",
        system_name="mirrors",
        pwm_capable=True,
    )
    _add(
        "zone_body",
        "hs_5a",
        "mirror_heater_right",
        load_type="ptc",
        system_cluster="body_comfort",
        system_name="mirrors",
        pwm_capable=True,
    )
    _add(
        "zone_body",
        "hs_10a",
        "door_lock_left",
        load_type="motor",
        inrush_factor=4.0,
        inrush_duration_ms=40.0,
        system_cluster="body_comfort",
        system_name="central_locking",
        driver_type="h_bridge",
    )
    _add(
        "zone_body",
        "hs_10a",
        "door_lock_right",
        load_type="motor",
        inrush_factor=4.0,
        inrush_duration_ms=40.0,
        system_cluster="body_comfort",
        system_name="central_locking",
        driver_type="h_bridge",
    )
    _add(
        "zone_body",
        "hs_15a",
        "window_left_front",
        load_type="motor",
        inrush_factor=5.0,
        inrush_duration_ms=50.0,
        system_cluster="body_comfort",
        system_name="power_windows",
        driver_type="h_bridge",
    )
    _add(
        "zone_body",
        "hs_15a",
        "window_right_front",
        load_type="motor",
        inrush_factor=5.0,
        inrush_duration_ms=50.0,
        system_cluster="body_comfort",
        system_name="power_windows",
        driver_type="h_bridge",
    )
    _add(
        "zone_body",
        "hs_15a",
        "window_left_rear",
        load_type="motor",
        inrush_factor=5.0,
        inrush_duration_ms=50.0,
        system_cluster="body_comfort",
        system_name="power_windows",
        driver_type="h_bridge",
    )
    _add(
        "zone_body",
        "hs_15a",
        "window_right_rear",
        load_type="motor",
        inrush_factor=5.0,
        inrush_duration_ms=50.0,
        system_cluster="body_comfort",
        system_name="power_windows",
        driver_type="h_bridge",
    )
    _add(
        "zone_body",
        "hs_30a",
        "seat_heater_left",
        load_type="ptc",
        connected_loads=["seat_heater_left", "lumbar_support_left"],
        system_cluster="body_comfort",
        system_name="seating",
        pwm_capable=True,
    )
    _add(
        "zone_body",
        "hs_30a",
        "seat_heater_right",
        load_type="ptc",
        connected_loads=["seat_heater_right", "lumbar_support_right"],
        system_cluster="body_comfort",
        system_name="seating",
        pwm_capable=True,
    )

    # ── Front zone (14 channels) ─────────────────────────────────
    _add(
        "zone_front",
        "hs_10a",
        "headlamp_low_left",
        system_cluster="exterior_lighting",
        system_name="front_lighting",
        pwm_capable=True,
    )
    _add(
        "zone_front",
        "hs_10a",
        "headlamp_low_right",
        system_cluster="exterior_lighting",
        system_name="front_lighting",
        pwm_capable=True,
    )
    _add(
        "zone_front",
        "hs_10a",
        "headlamp_high_left",
        system_cluster="exterior_lighting",
        system_name="front_lighting",
    )
    _add(
        "zone_front",
        "hs_10a",
        "headlamp_high_right",
        system_cluster="exterior_lighting",
        system_name="front_lighting",
    )
    _add(
        "zone_front",
        "hs_5a",
        "drl_left",
        connected_loads=["drl_left", "turn_signal_left"],
        system_cluster="exterior_lighting",
        system_name="front_lighting",
        pwm_capable=True,
        power_class="always_on",
    )
    _add(
        "zone_front",
        "hs_5a",
        "drl_right",
        connected_loads=["drl_right", "turn_signal_right"],
        system_cluster="exterior_lighting",
        system_name="front_lighting",
        pwm_capable=True,
        power_class="always_on",
    )
    _add(
        "zone_front",
        "hs_5a",
        "fog_light_left",
        system_cluster="exterior_lighting",
        system_name="front_lighting",
    )
    _add(
        "zone_front",
        "hs_5a",
        "fog_light_right",
        system_cluster="exterior_lighting",
        system_name="front_lighting",
    )
    _add(
        "zone_front",
        "hs_15a",
        "wiper_front",
        load_type="motor",
        inrush_factor=5.0,
        inrush_duration_ms=60.0,
        system_cluster="driver_assist",
        system_name="wipers",
    )
    _add(
        "zone_front",
        "hs_5a",
        "washer_pump",
        load_type="motor",
        inrush_factor=3.0,
        inrush_duration_ms=30.0,
        system_cluster="driver_assist",
        system_name="wipers",
    )
    _add(
        "zone_front",
        "hs_10a",
        "horn",
        load_type="inductive",
        system_cluster="driver_assist",
        system_name="signalling",
        power_class="always_on",
    )
    _add(
        "zone_front",
        "hs_2a",
        "rain_sensor",
        system_cluster="driver_assist",
        system_name="sensors",
        load_type="capacitive",
    )
    _add(
        "zone_front",
        "hs_5a",
        "adas_camera_power",
        connected_loads=["front_camera", "radar_sensor"],
        system_cluster="adas",
        system_name="perception",
        load_type="capacitive",
    )
    _add(
        "zone_front",
        "hs_2a",
        "adas_lidar_power",
        system_cluster="adas",
        system_name="perception",
        load_type="capacitive",
    )

    # ── Rear zone (10 channels) ──────────────────────────────────
    _add(
        "zone_rear",
        "hs_5a",
        "tail_light_left",
        connected_loads=["tail_left", "brake_left"],
        system_cluster="exterior_lighting",
        system_name="rear_lighting",
        pwm_capable=True,
        power_class="always_on",
    )
    _add(
        "zone_rear",
        "hs_5a",
        "tail_light_right",
        connected_loads=["tail_right", "brake_right"],
        system_cluster="exterior_lighting",
        system_name="rear_lighting",
        pwm_capable=True,
        power_class="always_on",
    )
    _add(
        "zone_rear",
        "hs_2a",
        "chmsl",
        system_cluster="exterior_lighting",
        system_name="rear_lighting",
        power_class="always_on",
    )
    _add(
        "zone_rear",
        "hs_2a",
        "license_plate_light",
        system_cluster="exterior_lighting",
        system_name="rear_lighting",
    )
    _add(
        "zone_rear",
        "hs_30a",
        "rear_defroster",
        load_type="ptc",
        system_cluster="body_comfort",
        system_name="climate",
        pwm_capable=True,
    )
    _add(
        "zone_rear",
        "hs_5a",
        "rear_wiper",
        load_type="motor",
        inrush_factor=4.0,
        inrush_duration_ms=40.0,
        system_cluster="driver_assist",
        system_name="wipers",
    )
    _add(
        "zone_rear",
        "ls_15a",
        "trunk_latch",
        load_type="motor",
        inrush_factor=4.0,
        inrush_duration_ms=50.0,
        system_cluster="body_comfort",
        system_name="closure",
        driver_type="low_side",
    )
    _add(
        "zone_rear",
        "hs_20a",
        "liftgate_motor",
        load_type="motor",
        inrush_factor=6.0,
        inrush_duration_ms=80.0,
        system_cluster="body_comfort",
        system_name="closure",
        driver_type="h_bridge",
    )
    _add(
        "zone_rear",
        "hs_2a",
        "parking_sensor_left",
        system_cluster="adas",
        system_name="parking",
        load_type="capacitive",
    )
    _add(
        "zone_rear",
        "hs_2a",
        "parking_sensor_right",
        system_cluster="adas",
        system_name="parking",
        load_type="capacitive",
    )

    # ── Underhood zone (12 channels) ─────────────────────────────
    _add(
        "zone_underhood",
        "hs_50a",
        "engine_fan",
        load_type="motor",
        inrush_factor=6.0,
        inrush_duration_ms=100.0,
        system_cluster="powertrain",
        system_name="engine_cooling",
        pwm_capable=True,
    )
    _add(
        "zone_underhood",
        "hs_30a",
        "fuel_pump",
        load_type="motor",
        inrush_factor=4.0,
        inrush_duration_ms=60.0,
        system_cluster="powertrain",
        system_name="fuel_system",
        pwm_capable=True,
    )
    _add(
        "zone_underhood",
        "hs_50a",
        "starter_relay",
        load_type="inductive",
        system_cluster="powertrain",
        system_name="starting",
        power_class="start",
    )
    _add(
        "zone_underhood",
        "hs_50a",
        "hvac_blower",
        load_type="motor",
        inrush_factor=5.0,
        inrush_duration_ms=80.0,
        system_cluster="body_comfort",
        system_name="climate",
        pwm_capable=True,
    )
    _add(
        "zone_underhood",
        "hs_20a",
        "hvac_compressor",
        load_type="motor",
        inrush_factor=4.0,
        inrush_duration_ms=60.0,
        connected_loads=["ac_compressor_clutch"],
        system_cluster="body_comfort",
        system_name="climate",
    )
    _add(
        "zone_underhood",
        "hs_15a",
        "coolant_pump",
        load_type="motor",
        inrush_factor=3.0,
        inrush_duration_ms=40.0,
        system_cluster="powertrain",
        system_name="engine_cooling",
        pwm_capable=True,
    )
    _add(
        "zone_underhood",
        "hs_10a",
        "throttle_body",
        load_type="motor",
        system_cluster="powertrain",
        system_name="engine_control",
        driver_type="h_bridge",
    )
    _add(
        "zone_underhood",
        "hs_5a",
        "egr_valve",
        load_type="inductive",
        system_cluster="powertrain",
        system_name="emissions",
        driver_type="half_bridge",
    )
    _add(
        "zone_underhood",
        "hs_5a",
        "purge_valve",
        load_type="inductive",
        system_cluster="powertrain",
        system_name="emissions",
        driver_type="half_bridge",
    )
    _add(
        "zone_underhood",
        "hs_5a",
        "o2_sensor_heater_upstream",
        load_type="ptc",
        system_cluster="powertrain",
        system_name="emissions",
        pwm_capable=True,
    )
    _add(
        "zone_underhood",
        "hs_5a",
        "o2_sensor_heater_downstream",
        load_type="ptc",
        system_cluster="powertrain",
        system_name="emissions",
        pwm_capable=True,
    )
    _add(
        "zone_underhood",
        "hs_2a",
        "under_hood_light",
        system_cluster="interior_lighting",
        system_name="utility_lights",
        power_class="always_on",
    )

    assert _ch == 52, f"Expected 52 channels, got {_ch}"
    return zones, specs
