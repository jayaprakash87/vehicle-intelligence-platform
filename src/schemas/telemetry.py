"""Telemetry schemas for the Vehicle Intelligence Platform.

This product detects and predicts eFuse faults in automotive electrical
systems by analysing current, voltage, and temperature telemetry from
Zone Controllers.

Data flows through three stages, each with a dedicated schema:

    TelemetryRecord  →  DerivedFeatures  →  InferenceResult
     (raw samples)      (rolling stats)     (fault verdict)

Supporting schemas:
  ChannelMeta      Per-channel static config (load profile, IC thresholds)
  EFuseProfile     IC-family electrical template (catalog look-up)
  ZoneController   Physical ECU hosting the eFuse ICs
  FaultInjection   Fault definition for simulation scenarios
  EventLabel       Ground-truth label for supervised evaluation

Validation happens at system boundaries (ingestion, inference output).
Internal hot-path code uses dicts/DataFrames for speed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_serializer


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DeviceStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    FAULT = "fault"
    UNKNOWN = "unknown"


class ProtectionEvent(str, Enum):
    """Protection mechanism reported by the eFuse IC / CDD.

    Real eFuse ICs expose a status register that tells the CDD exactly
    which protection mechanism fired. This information is essential for
    fault classification — the same trip_flag=True has very different
    implications depending on whether it was a 10µs short-circuit
    comparator or a slow I²t energy integral.
    """

    NONE = "none"  # No protection event active
    SCP = "scp"  # Short-circuit protection (instantaneous comparator)
    I2T = "i2t"  # I²t / F(i,t) energy-integral overcurrent trip
    THERMAL_SHUTDOWN = "thermal_shutdown"  # Junction temperature exceeded limit
    LATCH_OFF = "latch_off"  # Max retries exhausted — channel locked off


class FaultType(str, Enum):
    NONE = "none"
    OVERLOAD_SPIKE = "overload_spike"
    INTERMITTENT_OVERLOAD = "intermittent_overload"
    VOLTAGE_SAG = "voltage_sag"
    THERMAL_DRIFT = "thermal_drift"
    NOISY_SENSOR = "noisy_sensor"
    DROPPED_PACKET = "dropped_packet"
    GRADUAL_DEGRADATION = "gradual_degradation"


class SourceProtocol(str, Enum):
    """Telemetry source protocol — determines transport + typical sample rates."""

    CAN = "can"  # Production vehicles — 20-100 ms typical
    XCP = "xcp"  # Test vehicles — 10/50 ms DAQ rasters
    REPLAY = "replay"  # Offline replay from recorded files


class EFuseFamily(str, Enum):
    """eFuse IC families aligned to production vehicle programs.

    Covers Infineon PROFET+2 (BTS70xx), TLE multi-channel, BTS high-current,
    ST VIPower high-side (VN), dual (VND), H-bridge (VNH), low-side (VNL),
    and a CUSTOM entry for proprietary ASICs.

    Naming: {vendor}_{topology}_{current_class}
      INF = Infineon, ST = ST Microelectronics
      HS = high-side, LS = low-side, HB = H-bridge, MULTI = multi-channel
    """

    # Infineon PROFET+2 — BTS70xx, Rdson-parametric single/dual high-side
    INF_HS_2A = "inf_hs_2a"  # BTS7040-1EPA — 40mΩ, ~2.8A, LEDs, sensors
    INF_HS_5A = "inf_hs_5a"  # BTS7020-2EPA — 20mΩ, ~5.5A, dome lights, indicators
    INF_HS_9A = "inf_hs_9a"  # BTS7012-1EPA — 12mΩ, ~9A, locks, medium loads
    INF_HS_11A = "inf_hs_11a"  # BTS7010-1EPA — 10mΩ, ~11A, headlamps, fog lights
    INF_HS_14A = "inf_hs_14a"  # BTS7008-1EPA — 8mΩ, ~14A, wipers, windows
    INF_HS_18A = "inf_hs_18a"  # BTS7006-1EPZ — 6mΩ, ~18A, seat adjust, defroster
    INF_HS_28A = "inf_hs_28a"  # BTS7004-1EPP — 4mΩ, ~28A, seat heaters, HVAC

    # Infineon multi-channel and high-current
    INF_MULTI_10A = "inf_multi_10a"  # TLE92104-232QX — 4ch smart switch, ≤10A/ch
    INF_HS_100A = "inf_hs_100a"  # BTS81000-SSGI-6ET — high-current PDU, ≤100A

    # ST VIPower single high-side
    ST_HS_14A = "st_hs_14a"  # VN7140AS — single HS, ~14A
    ST_HS_30A = "st_hs_30a"  # VN9E30F — single HS, ~30A
    ST_HS_50A = "st_hs_50a"  # VN7050AS — single HS, ~50A

    # ST VIPower dual high-side, H-bridge, low-side
    ST_DUAL_14A = "st_dual_14a"  # VND7140AJ — dual HS, ~14A/ch
    ST_HB_30A = "st_hb_30a"  # VNH9045AQTR — H-bridge motor driver, ~30A
    ST_LS_50A = "st_ls_50a"  # VNL5050S5-E — low-side, ~50A

    # Custom / ASIC — user-defined eFuse with user-provided parameters
    CUSTOM = "custom"


class SafetyLevel(str, Enum):
    """Functional safety compliance level (ISO 26262)."""

    QM = "qm"  # Quality Management (no ASIL)
    ASIL_A = "asil_a"
    ASIL_B = "asil_b"
    ASIL_C = "asil_c"
    ASIL_D = "asil_d"


class DriverType(str, Enum):
    """Switch topology of the eFuse / power driver IC.

    Affects current measurement polarity, ground-offset behaviour,
    and which fault signatures are physically possible.
    """

    HIGH_SIDE = "high_side"  # Battery → switch → load → GND (most common)
    LOW_SIDE = "low_side"  # Battery → load → switch → GND
    H_BRIDGE = "h_bridge"  # Bidirectional motor drive
    HALF_BRIDGE = "half_bridge"  # Push-pull (coils, injectors)


class PowerClass(str, Enum):
    """Power rail / ignition-state dependency of a channel.

    Determines when the channel is expected to be energised:
      - ALWAYS_ON: permanent battery (KL30) — clock, hazard, memory keep-alive
      - IGNITION:  key-on supply (KL15) — most body/chassis functions
      - ACCESSORY: accessory rail (KLR) — radio, windows after key-off
      - START:     starter rail (KL50) — only during crank
    """

    ALWAYS_ON = "always_on"  # KL30 — permanent battery
    IGNITION = "ignition"  # KL15 — powered when ignition on
    ACCESSORY = "accessory"  # KLR  — powered in accessory mode
    START = "start"  # KL50 — powered only during crank


# ---------------------------------------------------------------------------
# eFuse electrical profile — template for a given IC type
# ---------------------------------------------------------------------------


class EFuseProfile(BaseModel):
    """Electrical + thermal template for an eFuse IC family.

    Captures the parameters that shape telemetry simulation and anomaly
    detection: current ratings, thermal model, ADC resolution, and
    protection thresholds.

    Each profile represents a specific IC or IC family (e.g. Infineon
    BTS7006-1EPP, ST VND7140AJ).  The optional identification fields
    (ic_part_number, manufacturer) provide traceability to real datasheets.
    """

    efuse_family: EFuseFamily
    nominal_current_a: float
    max_current_a: float
    fuse_rating_a: float
    r_ds_on_ohm: float = Field(description="MOSFET on-resistance Ω")
    r_thermal_kw: float = Field(description="Thermal resistance °C/W")
    tau_thermal_s: float = Field(description="Thermal time constant s")
    cooldown_s: float = 1.0
    max_retries: int = 3
    load_type: str = "resistive"

    # --- IC identification (optional — for traceability to real datasheets) ---
    ic_part_number: str = Field(default="", description="IC part number, e.g. 'BTS7006-1EPP'")
    manufacturer: str = Field(
        default="", description="IC vendor, e.g. 'Infineon', 'STMicroelectronics'"
    )

    # --- ADC sensing ---
    current_adc_bits: int = Field(
        default=12, description="Current-sense ADC resolution (varies by IC: 10-16 bit)"
    )
    voltage_adc_bits: int = Field(
        default=10, description="Voltage-sense ADC resolution (typ. 10-bit)"
    )

    # --- F(i,t) overcurrent protection ---
    # Many modern eFuse ICs integrate I²·t energy and trip when the
    # accumulated energy exceeds a threshold.  This is more realistic
    # than a simple peak-current comparator.
    fit_threshold_a2s: float = Field(
        default=0.0,
        description="F(i,t) trip threshold in A²·s.  0 = auto-derive from fuse_rating_a² × 0.01",
    )
    short_circuit_threshold_a: float = Field(
        default=0.0,
        description="Instantaneous short-circuit trip (A).  0 = auto as 3× max_current_a",
    )
    thermal_shutdown_c: float = Field(
        default=150.0,
        description="Junction temperature for thermal shutdown (°C). Typical: 150–175°C.",
    )

    # --- ISENSE current-sense chain ---
    # PROFET+2 / VIPower ICs expose an ILIS pin: I_ILIS = I_load / k_ILIS.
    # The CDD places an external R_ILIS to convert I_ILIS → V_ILIS → ADC.
    # Two independent error sources determine how far I_reported deviates from I_load:
    #   1. R_ILIS manufacturing tolerance (±20 % typical for 1 kΩ-class resistors)
    #   2. k_ILIS temperature coefficient (~-500 ppm/°C) and unit variation (±15 %)
    k_ilis: float = Field(
        default=5000.0,
        description="Current-sense ratio k_ILIS = I_load / I_ILIS (typ. 1 500–8 500 for PROFET+2)",
    )
    k_ilis_tempco_ppm_c: float = Field(
        default=-500.0,
        description="Temperature coefficient of k_ILIS in ppm/°C (typ. −400 to −600 ppm/°C)",
    )
    r_ilis_ohm: float = Field(
        default=1200.0,
        description="Nominal ISENSE sense resistor R_ILIS in Ω (board-level, OEM-selected)",
    )
    r_ilis_tolerance: float = Field(
        default=0.20,
        description="R_ILIS manufacturing tolerance fraction ± (e.g. 0.20 = ±20 %)",
    )

    # --- Safety classification ---
    safety_level: SafetyLevel = Field(
        default=SafetyLevel.QM, description="ISO 26262 classification"
    )


# ---------------------------------------------------------------------------
# Zone Controller — physical ECU hosting eFuse ICs
# ---------------------------------------------------------------------------


class ZoneController(BaseModel):
    """Zone Controller — the physical ECU/gateway hosting eFuse ICs.

    A vehicle typically has 2-4 Zone Controllers (body, front, rear,
    underhood).  Each Zone Controller runs a CDD (Complex Device Driver)
    — the AUTOSAR software layer that reads eFuse IC registers via SPI,
    measures current/voltage/temperature, and passes the signals via
    CAN or LIN to the application layer.

    Architecture:
        eFuse IC (HW) → SPI → CDD (SW driver) → COM stack → CAN/LIN bus
    """

    zone_id: str = Field(description="Unique Zone Controller identifier, e.g. 'zone_body'")
    name: str = Field(default="", description="Human-readable name, e.g. 'Body Zone Controller'")
    location: str = Field(
        default="body",
        description="Physical location: body | front | rear | underhood",
    )
    bus_interface: SourceProtocol = Field(
        default=SourceProtocol.CAN,
        description="Communication bus this zone controller's CDD publishes on",
    )
    cdd_read_cycle_ms: float = Field(
        default=10.0,
        description="CDD SPI read cycle in ms — how often the driver polls eFuse registers",
    )


# ---------------------------------------------------------------------------
# Core telemetry record — one row per sample per channel
# ---------------------------------------------------------------------------


class TelemetryRecord(BaseModel):
    """Single telemetry sample from one eFuse / channel."""

    timestamp: datetime
    channel_id: str
    current_a: float = Field(description="Measured current in amps")
    voltage_v: float = Field(description="Measured voltage in volts")
    temperature_c: float = Field(description="Junction / board temperature °C")
    state_on_off: bool = Field(description="Channel power state")
    trip_flag: bool = Field(default=False, description="Over-current trip active")
    overload_flag: bool = Field(default=False, description="Overload condition detected")
    protection_event: ProtectionEvent = Field(
        default=ProtectionEvent.NONE,
        description="Which protection mechanism fired (SCP, I²t, thermal, latch-off)",
    )
    reset_counter: int = Field(default=0, ge=0, description="Cumulative reset count")
    pwm_duty_pct: float = Field(default=100.0, ge=0.0, le=100.0, description="PWM duty %")
    device_status: DeviceStatus = DeviceStatus.OK

    @field_serializer("timestamp")
    @classmethod
    def _serialize_timestamp(cls, v: datetime) -> str:
        return v.isoformat()


# ---------------------------------------------------------------------------
# Event label — attached to simulated data for supervised evaluation
# ---------------------------------------------------------------------------


class EventLabel(BaseModel):
    """Ground-truth label produced by the simulator for a time window."""

    timestamp: datetime
    channel_id: str
    fault_type: FaultType = FaultType.NONE
    severity: float = Field(default=0.0, ge=0.0, le=1.0, description="0=nominal, 1=critical")
    description: str = ""


# ---------------------------------------------------------------------------
# Channel metadata
# ---------------------------------------------------------------------------


class ChannelMeta(BaseModel):
    """Static metadata describing one eFuse / load channel.

    Each channel represents one output of a Zone Controller, connected
    to one or more vehicle loads through a specific eFuse IC type.
    The CDD (Complex Device Driver) on the Zone Controller reads this
    channel's eFuse registers via SPI and publishes the measured signals.
    """

    channel_id: str
    load_name: str = ""
    nominal_current_a: float = 5.0
    max_current_a: float = 20.0
    nominal_voltage_v: float = 13.5
    fuse_rating_a: float = 15.0

    # eFuse type + physical grouping
    efuse_family: EFuseFamily = Field(
        default=EFuseFamily.INF_HS_14A, description="eFuse IC family from catalog"
    )
    zone_id: str = Field(
        default="", description="Zone Controller this channel belongs to (empty = unassigned)"
    )
    connected_loads: list[str] = Field(
        default_factory=list,
        description="Vehicle systems connected to this channel, e.g. ['seat_heater_left', 'lumbar_support_left']",
    )

    # --- Vehicle system hierarchy (ZC IO → Cluster → System → Component) ---
    system_cluster: str = Field(
        default="",
        description="Top-level grouping, e.g. 'exterior_lighting', 'body_comfort', 'powertrain'",
    )
    system_name: str = Field(
        default="",
        description="Sub-system under cluster, e.g. 'front_lighting', 'seating', 'engine_cooling'",
    )

    # --- IO characteristics ---
    driver_type: DriverType = Field(
        default=DriverType.HIGH_SIDE,
        description="Switch topology — affects current polarity and measurement",
    )
    power_class: PowerClass = Field(
        default=PowerClass.IGNITION,
        description="Power rail — when channel is expected to be energised",
    )
    pwm_capable: bool = Field(
        default=False,
        description="Whether the IO supports PWM switching (dimming, flash patterns)",
    )

    # Sampling rate — per-channel override; 0 means use global default
    sample_interval_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Channel sample interval in ms. 0 = use scenario default.",
    )
    source_protocol: SourceProtocol = Field(
        default=SourceProtocol.CAN,
        description="Source protocol (can | xcp | replay)",
    )

    # Load transient profile
    load_type: str = Field(
        default="resistive",
        description="Load model: resistive | inductive | motor | ptc | capacitive",
    )
    inrush_factor: float = Field(
        default=1.0, ge=1.0, description="Turn-on current multiplier vs nominal"
    )
    inrush_duration_ms: float = Field(
        default=0.0, ge=0.0, description="Inrush transient duration in ms"
    )

    # Thermal model (first-order RC)
    r_ds_on_ohm: float = Field(default=0.010, description="eFuse MOSFET on-resistance Ω")
    r_thermal_kw: float = Field(
        default=40.0, description="Junction-to-ambient thermal resistance °C/W"
    )
    tau_thermal_s: float = Field(
        default=15.0, description="Thermal time constant (R_th × C_th) seconds"
    )
    t_ambient_c: float = Field(default=25.0, description="Ambient temperature °C")

    # Noise profile
    current_adc_bits: int = Field(
        default=12, ge=8, le=16, description="Current-sense ADC resolution"
    )
    voltage_adc_bits: int = Field(
        default=10, ge=8, le=16, description="Voltage-sense ADC resolution"
    )
    pink_noise_alpha: float = Field(
        default=1.0, ge=0.0, le=2.0, description="1/f^α noise exponent, 0=white 1=pink"
    )
    emi_amplitude_a: float = Field(default=0.05, ge=0.0, description="EMI spike amplitude in amps")

    # Protection behavior
    fit_threshold_a2s: float = Field(
        default=0.0, ge=0.0, description="F(i,t) energy threshold A²·s (0 = auto)"
    )
    short_circuit_threshold_a: float = Field(
        default=0.0, ge=0.0, description="Instantaneous SCP trip (A) (0 = auto)"
    )
    thermal_shutdown_c: float = Field(
        default=150.0, ge=50.0, description="Junction temp for thermal shutdown (°C)"
    )
    cooldown_s: float = Field(default=1.0, ge=0.0, description="Auto-retry delay after eFuse trip")
    max_retries: int = Field(default=3, ge=0, description="Max auto-retries before latch-off")

    # --- ISENSE current-sense chain ---
    k_ilis: float = Field(
        default=5000.0,
        description="k_ILIS current-sense ratio I_load / I_ILIS",
    )
    k_ilis_tempco_ppm_c: float = Field(
        default=-500.0,
        description="k_ILIS temperature coefficient ppm/°C",
    )
    r_ilis_ohm: float = Field(
        default=1200.0,
        description="R_ILIS sense resistor nominal value Ω",
    )
    r_ilis_tolerance: float = Field(
        default=0.20,
        description="R_ILIS manufacturing tolerance fraction ±",
    )


class FaultInjection(BaseModel):
    """Specifies a fault to inject during simulation."""

    channel_id: str
    fault_type: FaultType
    start_s: float = Field(description="Seconds from scenario start")
    duration_s: float = Field(default=5.0, description="How long the fault lasts")
    intensity: float = Field(default=0.7, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Derived features record
# ---------------------------------------------------------------------------


class DerivedFeatures(BaseModel):
    """Rolling / derived features computed from raw telemetry."""

    timestamp: datetime
    channel_id: str
    rolling_rms_current: float = 0.0
    rolling_mean_current: float = 0.0
    rolling_max_current: float = 0.0
    rolling_min_current: float = 0.0
    temperature_slope: float = 0.0
    spike_score: float = 0.0
    anomaly_score: float = 0.0
    trip_frequency: float = 0.0
    recovery_time_s: float = 0.0
    degradation_trend: float = 0.0


# ---------------------------------------------------------------------------
# Inference output
# ---------------------------------------------------------------------------


class InferenceResult(BaseModel):
    """Output of the inference pipeline for one evaluation window."""

    timestamp: datetime
    channel_id: str
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    predicted_fault: FaultType = FaultType.NONE
    fault_confidence: float = 0.0
    likely_causes: list[str] = Field(default_factory=list)
    recommended_action: str = ""


# ---------------------------------------------------------------------------
# Cycle health
# ---------------------------------------------------------------------------


class HealthBand(str, Enum):
    """Discrete health classification for a completed cycle."""

    NOMINAL = "nominal"
    MONITOR = "monitor"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class CycleSummary(BaseModel):
    """Compact summary of one completed cycle (ignition / drive / charge).

    Produced by the CycleAccumulator at cycle-close.  Designed to be small
    enough for a PNDS-style ring buffer on the ECU.
    """

    cycle_id: str
    cycle_type: str = "ignition"
    open_timestamp: datetime
    close_timestamp: datetime
    duration_s: float = 0.0

    # Counters
    anomaly_count: int = 0
    trip_count: int = 0
    retry_count: int = 0
    sample_count: int = 0

    # Peaks
    peak_current_a: float = 0.0
    peak_temperature_c: float = 0.0

    # Dwell (seconds spent above thresholds)
    high_load_dwell_s: float = 0.0
    high_temp_dwell_s: float = 0.0
    voltage_sag_dwell_s: float = 0.0

    # Dominant fault
    dominant_fault: FaultType = FaultType.NONE
    dominant_fault_confidence: float = 0.0

    # Scores
    cycle_stress: float = 0.0
    health_band: HealthBand = HealthBand.NOMINAL

    @field_serializer("open_timestamp", "close_timestamp")
    def _serialize_ts(self, v: datetime, _info) -> str:
        return v.isoformat() if isinstance(v, datetime) else str(v)


# ---------------------------------------------------------------------------
# Lifetime health
# ---------------------------------------------------------------------------


class TrendDirection(str, Enum):
    """Trend of lifetime health over recent cycles."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    WORSENING = "worsening"


class LoadHistogram(BaseModel):
    """Fixed-bin histogram counter for one load spectrum dimension.

    Each bin counts the number of completed cycles whose value fell
    into that range.  Edges define the upper bounds of each bin
    (the last bin captures everything above the second-to-last edge).

    Example for peak current with 8 bins:
        edges = [2, 5, 8, 12, 15, 20, 30]   (7 edges → 8 bins)
        counts = [1204, 3891, 2107, 845, 312, 97, 18, 3]

    Memory: 8 × uint32 ≈ 32 bytes per histogram — NvM-friendly.
    """

    name: str  # e.g. "peak_current_a"
    unit: str = ""  # e.g. "A", "°C", "ratio"
    edges: list[float]  # N-1 upper-bound edges → N bins
    counts: list[int] = Field(default_factory=list)  # one count per bin

    def model_post_init(self, __context) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.edges) + 1)

    def bin_index(self, value: float) -> int:
        """Return the bin index for a given value."""
        for i, edge in enumerate(self.edges):
            if value < edge:
                return i
        return len(self.edges)

    def record(self, value: float) -> None:
        """Increment the bin corresponding to *value*."""
        self.counts[self.bin_index(value)] += 1

    @property
    def total(self) -> int:
        return sum(self.counts)

    def upper_fraction(self, top_n_bins: int = 2) -> float:
        """Fraction of total counts in the top *top_n_bins* bins."""
        t = self.total
        if t == 0:
            return 0.0
        return sum(self.counts[-top_n_bins:]) / t


class LifetimeHealthState(BaseModel):
    """Histogram-based lifetime state — the automotive-correct approach.

    Core state is 6 load-spectrum histograms (8 bins each ≈ 192 bytes).
    Health score and trend are **derived** from histogram shape, not
    stored independently.

    Updated once per completed cycle — not per telemetry sample.
    """

    # Identity
    cycles_ingested: int = 0

    # --- Load-spectrum histograms (primary state) ---
    peak_current_hist: LoadHistogram = Field(
        default_factory=lambda: LoadHistogram(
            name="peak_current_a", unit="A",
            edges=[2.0, 5.0, 8.0, 12.0, 15.0, 20.0, 30.0],
        )
    )
    peak_temperature_hist: LoadHistogram = Field(
        default_factory=lambda: LoadHistogram(
            name="peak_temperature_c", unit="°C",
            edges=[40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0],
        )
    )
    cycle_stress_hist: LoadHistogram = Field(
        default_factory=lambda: LoadHistogram(
            name="cycle_stress", unit="ratio",
            edges=[0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80],
        )
    )
    trips_per_cycle_hist: LoadHistogram = Field(
        default_factory=lambda: LoadHistogram(
            name="trips_per_cycle", unit="count",
            edges=[1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
        )
    )
    retries_per_cycle_hist: LoadHistogram = Field(
        default_factory=lambda: LoadHistogram(
            name="retries_per_cycle", unit="count",
            edges=[1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
        )
    )
    thermal_dwell_frac_hist: LoadHistogram = Field(
        default_factory=lambda: LoadHistogram(
            name="thermal_dwell_fraction", unit="ratio",
            edges=[0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75],
        )
    )

    # --- Derived (recomputed on each ingest, not primary state) ---
    health_score: float = 1.0  # 1.0 = perfect, 0.0 = critical
    health_band: HealthBand = HealthBand.NOMINAL
    trend: TrendDirection = TrendDirection.STABLE

    # Snapshot
    last_cycle_id: str = ""
    last_update_timestamp: datetime | None = None

    @field_serializer("last_update_timestamp")
    def _serialize_ts(self, v: datetime | None, _info) -> str | None:
        if v is None:
            return None
        return v.isoformat() if isinstance(v, datetime) else str(v)
