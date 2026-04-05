"""Synthetic telemetry generator with physically grounded signal models.

Produces time-series DataFrames that approximate real eFuse telemetry.
Signal realism features:
  - Composite noise (1/f + quantization + thermal + EMI)
  - First-order RC junction temperature model
  - Shared bus voltage with alternator ripple
  - Load-type-specific turn-on transients
  - Realistic fault waveforms (exponential rise/fall, oscillations)
  - eFuse protection response (trip → off → cooldown → retry)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.config.models import SimulationConfig
from src.schemas.telemetry import (
    ChannelMeta,
    DeviceStatus,
    FaultInjection,
    FaultType,
    PowerClass,
    PowerState,
    ProtectionEvent,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


class TelemetryGenerator:
    """Generates synthetic eFuse-style telemetry for configured channels."""

    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (telemetry_df, labels_df) for the full scenario."""
        t0 = datetime.now(tz=timezone.utc)

        # Determine per-channel intervals (ms); fall back to global default
        ch_intervals = {}
        for ch in self.cfg.channels:
            if ch.sample_interval_ms > 0:
                ch_intervals[ch.channel_id] = ch.sample_interval_ms
            else:
                ch_intervals[ch.channel_id] = self.cfg.sample_interval_ms

        # Bus voltage at the finest granularity across all channels
        finest_ms = min(ch_intervals.values())
        finest_s = finest_ms / 1000
        n_finest = int(self.cfg.duration_s / finest_s)
        bus_voltage_fine = self._generate_bus_voltage(n_finest, finest_s)
        np.arange(n_finest) * finest_s  # seconds from t0

        # Power-state timeline: build per-sample array at finest resolution
        power_states_fine = self._build_power_state_array(n_finest, finest_s)

        all_telem: list[pd.DataFrame] = []
        all_labels: list[pd.DataFrame] = []

        for ch in self.cfg.channels:
            interval_ms = ch_intervals[ch.channel_id]
            interval_s = interval_ms / 1000
            n_samples = int(self.cfg.duration_s / interval_s)
            timestamps = [t0 + timedelta(milliseconds=i * interval_ms) for i in range(n_samples)]

            # Downsample bus voltage to this channel's rate
            if interval_ms == finest_ms:
                bus_voltage = bus_voltage_fine[:n_samples]
            else:
                step = max(int(interval_ms / finest_ms), 1)
                bus_voltage = bus_voltage_fine[::step][:n_samples]
                # Pad if rounding left us short
                if len(bus_voltage) < n_samples:
                    bus_voltage = np.pad(
                        bus_voltage, (0, n_samples - len(bus_voltage)), mode="edge"
                    )

            ch_faults = [f for f in self.cfg.fault_injections if f.channel_id == ch.channel_id]

            # Downsample power states to this channel's rate (list, not numpy array)
            if interval_ms == finest_ms:
                power_states = power_states_fine[:n_samples]
            else:
                step = max(int(interval_ms / finest_ms), 1)
                power_states = power_states_fine[::step][:n_samples]
                # Pad with last state if rounding left us short
                while len(power_states) < n_samples:
                    power_states = power_states + [power_states[-1]]

            ch_df, label_df = self._generate_channel(
                timestamps, ch, ch_faults, bus_voltage, power_states, interval_s
            )
            all_telem.append(ch_df)
            all_labels.append(label_df)

        telem_df = pd.concat(all_telem, ignore_index=True) if all_telem else pd.DataFrame()
        labels_df = pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame()

        # --- Multi-channel die thermal coupling ---
        # Channels sharing the same die_id exchange heat through the substrate.
        # For each die group, the steady-state ΔT of every other channel on the
        # same die is injected into this channel scaled by thermal_coupling_coeff.
        # ΔT_neighbour = (T_mean - T_ambient) is the heat source;
        # We add k × ΔT_neighbour to the target channel's temperature array.
        if not telem_df.empty and "temperature_c" in telem_df.columns:
            telem_df = self._apply_die_thermal_coupling(telem_df)

        log.info(
            "Generated %d telemetry rows, %d labels across %d channels",
            len(telem_df),
            len(labels_df),
            len(self.cfg.channels),
        )
        return telem_df, labels_df

    # ------------------------------------------------------------------
    # Bus voltage generation (shared across channels)
    # ------------------------------------------------------------------

    def _build_power_state_array(self, n: int, interval_s: float) -> list[PowerState]:
        """Return a list of PowerState values, one per sample.

        Uses a Python list (not numpy) so str-Enum instances are preserved
        exactly — numpy object arrays coerce str subclasses to plain str in
        Python ≥ 3.11, breaking enum equality comparisons.

        If no power_state_events are configured, every sample is ACTIVE.
        Events are applied in time order; the state before the first event
        defaults to ACTIVE.
        """
        states: list[PowerState] = [PowerState.ACTIVE] * n
        events = sorted(self.cfg.power_state_events, key=lambda e: e.time_s)
        for i, event in enumerate(events):
            start_idx = max(0, min(int(event.time_s / interval_s), n))
            end_idx = int(events[i + 1].time_s / interval_s) if i + 1 < len(events) else n
            end_idx = max(start_idx, min(end_idx, n))
            for j in range(start_idx, end_idx):
                states[j] = event.state
        return states

    def _apply_die_thermal_coupling(self, telem_df: pd.DataFrame) -> pd.DataFrame:
        """Inject cross-channel die heat into co-located channels.

        For each die group (channels with same non-empty die_id):
          - Compute each channel's per-sample ΔT above ambient
          - Sum-and-scale the neighbour contributions: ΔT_coupled = k × Σ ΔT_j (j ≠ i)
          - Add to the target channel's temperature_c array, clipped to thermal_shutdown_c + 10
        """
        # Build a lookup: channel_id → ChannelMeta
        ch_map = {ch.channel_id: ch for ch in self.cfg.channels}

        # Group channels by die_id (skip empty / isolated channels)
        die_groups: dict[str, list[str]] = {}
        for ch in self.cfg.channels:
            if ch.die_id:
                die_groups.setdefault(ch.die_id, []).append(ch.channel_id)

        # Only process dies with ≥ 2 channels
        for die_id, members in die_groups.items():
            if len(members) < 2:
                continue

            # Extract temperature arrays for all members (aligned by position in DataFrame)
            # Channels may have different sample rates — work channel-by-channel
            for ch_id in members:
                ch = ch_map[ch_id]
                target_mask = telem_df["channel_id"] == ch_id
                target_temps = telem_df.loc[target_mask, "temperature_c"].to_numpy(dtype=float)
                n_target = len(target_temps)

                coupled_delta = np.zeros(n_target, dtype=float)

                for nbr_id in members:
                    if nbr_id == ch_id:
                        continue
                    nbr = ch_map[nbr_id]
                    nbr_mask = telem_df["channel_id"] == nbr_id
                    nbr_temps = telem_df.loc[nbr_mask, "temperature_c"].to_numpy(dtype=float)
                    n_nbr = len(nbr_temps)

                    # ΔT of neighbour above its own ambient
                    nbr_ambient = nbr.t_ambient_c
                    nbr_delta = np.clip(nbr_temps - nbr_ambient, 0.0, None)

                    # Resample to target length (nearest-neighbour; handles different rates)
                    if n_nbr != n_target:
                        idx = np.round(np.linspace(0, n_nbr - 1, n_target)).astype(int)
                        nbr_delta = nbr_delta[idx]

                    # Average coupling coefficient between the two channels
                    k = (ch.thermal_coupling_coeff + nbr.thermal_coupling_coeff) / 2.0
                    coupled_delta += k * nbr_delta

                # Apply coupling: raise target temperature by the summed neighbour contribution
                new_temps = target_temps + coupled_delta
                shutdown_limit = ch.thermal_shutdown_c + 10.0
                new_temps = np.clip(new_temps, -40.0, shutdown_limit)
                telem_df.loc[target_mask, "temperature_c"] = new_temps

        return telem_df

    # ------------------------------------------------------------------
    # Bus voltage generation
    # ------------------------------------------------------------------

    def _generate_bus_voltage(self, n: int, interval_s: float) -> np.ndarray:
        """Generate a shared bus voltage with alternator ripple and minor sag events.

        Models: 13.5V nominal + 100mV alternator ripple at ~50Hz equivalent
        plus slow drift from battery charge state.
        """
        t = np.arange(n) * interval_s
        # Alternator ripple (rectified AC, ~50 Hz aliased to sample rate)
        ripple_freq = 50.0  # Hz — 6-cylinder engine at ~1000 RPM
        ripple = 0.1 * np.sin(2 * np.pi * ripple_freq * t)
        # Slow battery charge drift
        drift = 0.2 * np.sin(2 * np.pi * t / max(t[-1], 1.0) * 0.3)
        bus = 13.5 + ripple + drift + self.rng.normal(0, 0.02, n)
        return bus

    # ------------------------------------------------------------------
    # Noise generation
    # ------------------------------------------------------------------

    def _composite_noise(self, n: int, ch: ChannelMeta) -> np.ndarray:
        """Generate composite noise: 1/f^α + quantization + thermal + EMI spikes."""
        noise = np.zeros(n)

        # 1/f (pink) noise via spectral shaping
        if ch.pink_noise_alpha > 0:
            white = self.rng.normal(0, 0.1, n)
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n, d=1.0)
            freqs[0] = 1.0  # avoid div/0 at DC
            fft *= 1.0 / (freqs ** (ch.pink_noise_alpha / 2))
            pink = np.fft.irfft(fft, n=n)
            # Normalize to ~0.1A std — guard against n=1 where std is always 0
            std = np.std(pink)
            if std > 1e-10:
                pink *= 0.1 / std
            else:
                pink[:] = 0.0  # single-sample window: skip pink noise
            noise += pink

        # ADC quantization noise (current-sense ADC)
        adc_range = ch.max_current_a * 1.5  # full-scale range
        lsb = adc_range / (2**ch.current_adc_bits)
        quant_noise = self.rng.uniform(-lsb / 2, lsb / 2, n)
        noise += quant_noise

        # Thermal noise (proportional to sqrt of temperature baseline)
        thermal_factor = np.sqrt(max(ch.t_ambient_c + 273.15, 1.0) / 298.15)
        noise += self.rng.normal(0, 0.02 * thermal_factor, n)

        # Sporadic EMI spikes
        emi_prob = 0.005  # ~0.5% of samples
        emi_mask = self.rng.random(n) < emi_prob
        emi_sign = self.rng.choice([-1, 1], size=n)
        noise += emi_mask * emi_sign * ch.emi_amplitude_a

        return noise

    # ------------------------------------------------------------------
    # Thermal model
    # ------------------------------------------------------------------

    def _rc_thermal(self, current: np.ndarray, ch: ChannelMeta, interval_s: float) -> np.ndarray:
        """First-order RC thermal model: T_j = T_amb + P · R_th · (1 - e^(-t/τ)).

        Computed iteratively to handle dynamic power changes from load current.

        Rds,on is temperature-dependent (PROFET+2 / VIPower power-law):
            R_ds,on(T_j) = R_ds,on(25°C) × (T_j_K / 300)^n
        where T_j_K is junction temperature in Kelvin and n ≈ 2.3.

        This models the positive thermal feedback loop:
            more current → more heat → higher Rds,on → more P = I²·R → more heat.

        P = I² · R_ds,on(T_j)
        """
        n = len(current)
        temp = np.empty(n)
        t_j = ch.t_ambient_c  # initial junction temp = ambient (°C)
        alpha = interval_s / ch.tau_thermal_s  # discrete time constant
        t_ref_k = 300.0  # reference temperature in Kelvin (≈ 27°C)
        exp = ch.rds_on_tempco_exp

        for i in range(n):
            # Temperature-dependent Rds,on — cap at ambient (IC is off above T_sd)
            t_j_safe = min(t_j, ch.thermal_shutdown_c) if np.isfinite(t_j) else ch.t_ambient_c
            t_j_k = max(t_j_safe + 273.15, 1.0)  # guard against non-positive K
            r_ds_on_t = ch.r_ds_on_ohm * (t_j_k / t_ref_k) ** exp
            power_w = current[i] ** 2 * r_ds_on_t
            t_steady = ch.t_ambient_c + power_w * ch.r_thermal_kw
            # Exponential approach to steady state
            t_j = t_j + alpha * (t_steady - t_j)
            temp[i] = t_j

        return np.clip(temp, -40, 150)

    # ------------------------------------------------------------------
    # ISENSE sensing chain
    # ------------------------------------------------------------------

    def _apply_isense_chain(
        self,
        current: np.ndarray,
        temp: np.ndarray,
        ch: ChannelMeta,
    ) -> np.ndarray:
        """Apply ISENSE sensing-chain errors to the true load current.

        Models the PROFET+2 / VIPower ILIS output as read by the CDD ADC:

            I_reported = I_load × (1 + δ_r) × (1 + δ_k) × (1 + α_k × (T_j − 25))

        where:
          δ_r ~ U(±r_ilis_tolerance)  — R_ILIS manufacturing tolerance, frozen
          δ_k ~ U(±0.15)              — k_ILIS unit-to-unit variation, frozen
          α_k = k_ilis_tempco_ppm_c × 1e-6  — dynamic temperature coefficient

        Both frozen offsets are sampled once per call (i.e. per channel per
        simulation run), matching a real unit that ships with fixed but unknown
        component offsets.  NaN values (dropped-packet samples) propagate
        unchanged.
        """
        # Frozen per-channel manufacturing scatter (one draw per simulation run)
        delta_r = self.rng.uniform(-ch.r_ilis_tolerance, ch.r_ilis_tolerance)
        delta_k = self.rng.uniform(-0.15, 0.15)  # ±15 % k_ILIS unit variation
        k_fixed = (1.0 + delta_r) * (1.0 + delta_k)

        # Dynamic temperature coefficient (vectorised; NaN-safe)
        alpha_k = ch.k_ilis_tempco_ppm_c * 1e-6  # ppm/°C → 1/°C
        k_temp = 1.0 + alpha_k * (temp - 25.0)

        return current * k_fixed * k_temp

    # ------------------------------------------------------------------
    # Load transient
    # ------------------------------------------------------------------

    def _apply_inrush(self, current: np.ndarray, ch: ChannelMeta, interval_s: float) -> np.ndarray:
        """Apply load-type-specific turn-on transient to the first N samples."""
        if ch.inrush_factor <= 1.0 or ch.inrush_duration_ms <= 0:
            # Default resistive load types or no inrush configured
            if ch.load_type == "motor":
                # Motor: 5x inrush for 50ms if not explicitly configured
                n_inrush = max(int(0.050 / interval_s), 1)
                factor = 5.0
            elif ch.load_type == "inductive":
                n_inrush = max(int(0.020 / interval_s), 1)
                factor = 3.0
            elif ch.load_type == "ptc":
                # PTC thermistor: 2x cold-start for 500ms, decaying
                n_inrush = max(int(0.500 / interval_s), 1)
                factor = 2.0
            elif ch.load_type == "capacitive":
                # Capacitive: sharp spike (input caps charging), very fast decay
                # Typical: 8-10x for <10ms, RC-shaped
                n_inrush = max(int(0.010 / interval_s), 1)
                factor = 8.0
            else:
                return current  # resistive: no transient
        else:
            n_inrush = max(int(ch.inrush_duration_ms / 1000 / interval_s), 1)
            factor = ch.inrush_factor

        n_inrush = min(n_inrush, len(current))

        if ch.load_type == "capacitive" and ch.inrush_factor <= 1.0:
            # Capacitive loads: fast RC discharge shape (steeper than exponential)
            decay = np.exp(-5 * np.arange(n_inrush) / max(n_inrush, 1))
        else:
            # All other loads: standard exponential decay (~95% settled by end)
            decay = np.exp(-3 * np.arange(n_inrush) / max(n_inrush, 1))

        inrush_envelope = 1.0 + (factor - 1.0) * decay
        current[:n_inrush] *= inrush_envelope

        return current

    # ------------------------------------------------------------------
    # Fault waveform shaping
    # ------------------------------------------------------------------

    def _fault_envelope_rise_fall(
        self, n_fault: int, rise_frac: float = 0.15, fall_frac: float = 0.15
    ) -> np.ndarray:
        """Generate a trapezoidal envelope with exponential rise and fall."""
        env = np.ones(n_fault)
        n_rise = max(int(n_fault * rise_frac), 1)
        n_fall = max(int(n_fault * fall_frac), 1)
        # Exponential rise
        env[:n_rise] = 1 - np.exp(-3 * np.arange(n_rise) / n_rise)
        # Exponential fall
        env[-n_fall:] = np.exp(-3 * np.arange(n_fall) / n_fall)
        return env

    def _oscillating_envelope(
        self, n_fault: int, freq_cycles: float = 3.0, damping: float = 2.0
    ) -> np.ndarray:
        """Damped oscillation for intermittent faults."""
        t = np.linspace(0, 1, n_fault)
        osc = np.exp(-damping * t) * np.abs(np.sin(2 * np.pi * freq_cycles * t))
        return osc

    # ------------------------------------------------------------------
    # Protection response
    # ------------------------------------------------------------------

    def _apply_protection(
        self,
        current: np.ndarray,
        state: np.ndarray,
        trip: np.ndarray,
        protection_event: np.ndarray,
        reset_counter: np.ndarray,
        ch: ChannelMeta,
        fault_start: int,
        fault_end: int,
        interval_s: float,
    ) -> None:
        """Model eFuse F(i,t) energy-integral protection with fast SCP.

        Two protection mechanisms run in parallel:
          1. **Short-circuit (SCP)** — instantaneous comparator trip when
             current exceeds short_circuit_threshold_a.
          2. **F(i,t) overcurrent** — trips when cumulative ∫ I² dt exceeds
             fit_threshold_a2s.  This allows brief inrush spikes but catches
             sustained overloads.

        After trip: channel off → cooldown → retry → latch-off if max retries
        exceeded.  Tags each sample's protection_event with the specific
        mechanism that fired (SCP, I2T, or LATCH_OFF).
        Mutates arrays in-place.
        """
        # Resolve auto thresholds (0 means "derive from profile")
        fit_thresh = ch.fit_threshold_a2s
        if fit_thresh <= 0:
            fit_thresh = ch.fuse_rating_a**2 * 0.01  # sensible default

        scp_thresh = ch.short_circuit_threshold_a
        if scp_thresh <= 0:
            scp_thresh = ch.max_current_a * 3.0

        cooldown_samples = max(int(ch.cooldown_s / interval_s), 1)
        retries_done = 0
        i = fault_start
        energy = 0.0  # running ∫ I² dt accumulator

        while i < fault_end and retries_done <= ch.max_retries:
            # Scan forward looking for a trip condition
            trip_reason = ProtectionEvent.NONE
            scan_end = min(fault_end, i + cooldown_samples * 10)  # bounded scan
            for j in range(i, scan_end):
                i_abs = abs(current[j])

                # Fast SCP — immediate trip
                if i_abs >= scp_thresh:
                    i = j
                    trip_reason = ProtectionEvent.SCP
                    break

                # F(i,t) energy integration (only when above nominal)
                if i_abs > ch.nominal_current_a:
                    energy += (i_abs**2) * interval_s
                else:
                    # Drain energy slowly when below nominal (thermal dissipation)
                    energy = max(0.0, energy - (ch.nominal_current_a**2) * interval_s * 0.5)

                if energy >= fit_thresh:
                    i = j
                    trip_reason = ProtectionEvent.I2T
                    break

            if trip_reason == ProtectionEvent.NONE:
                break  # fault current not high enough to trip

            # --- Trip event ---
            trip[i:fault_end] = True
            state[i:fault_end] = False
            cooldown_end = min(i + cooldown_samples, fault_end)
            current[i:cooldown_end] = self.rng.normal(0.001, 0.0005, cooldown_end - i)
            state[i:cooldown_end] = False
            protection_event[i:cooldown_end] = trip_reason.value

            retries_done += 1
            reset_counter[cooldown_end:fault_end] = retries_done
            energy = 0.0  # reset accumulator after trip+cooldown

            # After cooldown, retry (auto-restart)
            i = cooldown_end
            if i < fault_end:
                state[i:fault_end] = True
                trip[i:fault_end] = False
                protection_event[i:fault_end] = ProtectionEvent.NONE.value

        # If max retries exhausted, latch off for remainder
        if retries_done > ch.max_retries and i < fault_end:
            current[i:fault_end] = self.rng.normal(0.0, 0.0005, fault_end - i)
            state[i:fault_end] = False
            trip[i:fault_end] = True
            protection_event[i:fault_end] = ProtectionEvent.LATCH_OFF.value

    # ------------------------------------------------------------------
    # Per-channel generation
    # ------------------------------------------------------------------

    def _generate_channel(
        self,
        timestamps: list[datetime],
        ch: ChannelMeta,
        faults: list[FaultInjection],
        bus_voltage: np.ndarray,
        power_states: list[PowerState],
        interval_s: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        n = len(timestamps)

        # --- Nominal baselines ---
        # Current: nominal + composite noise
        current = np.full(n, ch.nominal_current_a) + self._composite_noise(n, ch)

        # Apply load turn-on transient
        current = self._apply_inrush(current, ch, interval_s)

        # Voltage: derived from shared bus - resistive drop through harness + connectors
        harness_r = ch.harness_r_ohm + ch.connector_r_ohm
        voltage = bus_voltage - current * harness_r + self.rng.normal(0, 0.01, n)

        state = np.ones(n, dtype=bool)
        trip = np.zeros(n, dtype=bool)
        overload = np.zeros(n, dtype=bool)
        protection_event = np.full(n, ProtectionEvent.NONE.value, dtype=object)
        reset_counter = np.zeros(n, dtype=int)
        pwm = np.full(n, 100.0)
        status = np.full(n, DeviceStatus.OK.value, dtype=object)
        fault_active = np.full(n, FaultType.NONE.value, dtype=object)
        severity = np.zeros(n)

        # --- Apply power-state gating ---
        # Determine for each sample whether this channel is powered based on its
        # power_class and the scenario's power_state_events.
        #
        # Power-class × state mapping:
        #   ALWAYS_ON  (KL30)  → ON in all states; SLEEP = quiescent dark current
        #   IGNITION   (KL15)  → ON only in ACTIVE; off in SLEEP/ACCESSORY/CRANK
        #   ACCESSORY  (KLR)   → ON in ACTIVE + ACCESSORY; off in SLEEP/CRANK
        #   START      (KL50)  → ON only in CRANK; off everywhere else
        quiescent_a = ch.sleep_quiescent_ua * 1e-6  # µA → A
        prev_state: PowerState | None = None
        for i in range(n):
            ps: PowerState = power_states[i]

            # Determine if this channel is powered in this state
            if ch.power_class == PowerClass.ALWAYS_ON:
                powered = True
            elif ch.power_class == PowerClass.IGNITION:
                powered = ps == PowerState.ACTIVE
            elif ch.power_class == PowerClass.ACCESSORY:
                powered = ps in (PowerState.ACTIVE, PowerState.ACCESSORY)
            elif ch.power_class == PowerClass.START:
                powered = ps == PowerState.CRANK
            else:
                powered = True

            if not powered:
                # Gate off: near-zero leakage for all non-powered channels
                current[i] = self.rng.normal(0.00005, 0.00002)  # < 0.1 mA
                state[i] = False
                status[i] = DeviceStatus.OK.value  # off intentionally — not a fault

            else:
                # ALWAYS_ON in SLEEP: draw quiescent dark current (KL30 stays on but at standby level)
                if ch.power_class == PowerClass.ALWAYS_ON and ps == PowerState.SLEEP:
                    current[i] = quiescent_a + self.rng.normal(0, quiescent_a * 0.1)
                # Wake transition: apply inrush on first powered sample after unpowered
                if prev_state is not None and not (
                    (ch.power_class == PowerClass.ALWAYS_ON)
                    or (ch.power_class == PowerClass.IGNITION and prev_state == PowerState.ACTIVE)
                    or (ch.power_class == PowerClass.ACCESSORY and prev_state in (PowerState.ACTIVE, PowerState.ACCESSORY))
                    or (ch.power_class == PowerClass.START and prev_state == PowerState.CRANK)
                ):
                    # Came from an unpowered state — inject wake inrush window
                    inrush_samples = max(int(ch.wake_inrush_duration_ms / 1000 / interval_s), 1)
                    end_inrush = min(i + inrush_samples, n)
                    t_ramp = np.linspace(ch.wake_inrush_factor, 1.0, end_inrush - i)
                    current[i:end_inrush] = (
                        ch.nominal_current_a * t_ramp
                        + self._composite_noise(end_inrush - i, ch)
                    )

            prev_state = ps

        # --- Apply fault injections with realistic waveforms ---
        for fi in faults:
            start_idx = int(fi.start_s / interval_s)
            end_idx = min(start_idx + int(fi.duration_s / interval_s), n)
            n_fault = end_idx - start_idx
            if n_fault <= 0:
                continue
            sl = slice(start_idx, end_idx)

            match fi.fault_type:
                case FaultType.OVERLOAD_SPIKE:
                    # Exponential rise to overcurrent, then protection kicks in
                    envelope = self._fault_envelope_rise_fall(
                        n_fault, rise_frac=0.05, fall_frac=0.0
                    )
                    peak = ch.max_current_a * (1.0 + fi.intensity * 0.5)
                    current[sl] = peak * envelope + self._composite_noise(n_fault, ch) * 0.5
                    trip[sl] = True
                    overload[sl] = True
                    status[sl] = DeviceStatus.FAULT.value
                    # Protection response: trip → off → cooldown → retry
                    self._apply_protection(
                        current,
                        state,
                        trip,
                        protection_event,
                        reset_counter,
                        ch,
                        start_idx,
                        end_idx,
                        interval_s,
                    )

                case FaultType.INTERMITTENT_OVERLOAD:
                    # Damped oscillation — repeated overcurrent bursts
                    osc = self._oscillating_envelope(
                        n_fault, freq_cycles=4.0 * fi.intensity, damping=1.5
                    )
                    current[sl] = (
                        ch.nominal_current_a
                        + osc * (ch.max_current_a * 1.2 - ch.nominal_current_a) * fi.intensity
                    )
                    current[sl] += self._composite_noise(n_fault, ch) * 0.3
                    overload_mask = current[sl] > ch.fuse_rating_a * 0.9
                    overload[start_idx:end_idx] = overload_mask
                    status[sl] = np.where(
                        overload_mask, DeviceStatus.WARNING.value, DeviceStatus.OK.value
                    )

                case FaultType.VOLTAGE_SAG:
                    # Exponential voltage drop (like battery under heavy load)
                    envelope = self._fault_envelope_rise_fall(
                        n_fault, rise_frac=0.10, fall_frac=0.20
                    )
                    sag_depth = ch.nominal_voltage_v * 0.3 * fi.intensity
                    voltage[sl] = (
                        bus_voltage[sl] - sag_depth * envelope + self.rng.normal(0, 0.05, n_fault)
                    )
                    status[sl] = DeviceStatus.WARNING.value

                case FaultType.THERMAL_DRIFT:
                    # Gradually increasing current draw (simulating insulation breakdown)
                    ramp = np.linspace(1.0, 1.0 + 0.3 * fi.intensity, n_fault)
                    current[sl] *= ramp
                    # Temperature effect is handled by the RC thermal model below
                    status[sl] = DeviceStatus.WARNING.value

                case FaultType.NOISY_SENSOR:
                    # Burst noise with varying amplitude — not flat Gaussian
                    burst_envelope = self._oscillating_envelope(
                        n_fault, freq_cycles=6.0, damping=0.5
                    )
                    noise_scale = 2.0 * fi.intensity
                    current[sl] += burst_envelope * self.rng.normal(0, noise_scale, n_fault)
                    voltage[sl] += burst_envelope * self.rng.normal(0, noise_scale * 0.2, n_fault)

                case FaultType.DROPPED_PACKET:
                    drop_mask = self.rng.random(n_fault) < 0.4 * fi.intensity
                    current[start_idx:end_idx] = np.where(drop_mask, np.nan, current[sl])
                    voltage[start_idx:end_idx] = np.where(drop_mask, np.nan, voltage[sl])

                case FaultType.GRADUAL_DEGRADATION:
                    # Slow exponential ramp (aging load — insulation breakdown / draw increase)
                    t_norm = np.linspace(0, 1, n_fault)
                    deg = 1.0 + fi.intensity * 0.5 * (np.exp(2 * t_norm) - 1) / (np.e**2 - 1)
                    current[sl] *= deg
                    pwm[sl] = np.clip(100 - np.linspace(0, 30 * fi.intensity, n_fault), 0, 100)
                    status[sl] = DeviceStatus.WARNING.value

                case FaultType.CONNECTOR_AGING:
                    # Fretting corrosion / oxidation raises pin contact resistance over time.
                    # Fresh terminal ≈ 5–10 mΩ; corroded terminal ≈ 50–500 mΩ.
                    # Model: connector_r grows exponentially — R_c(t) = R_c0 × (1 + k × t_norm²)
                    # where k = intensity × 20 gives a factor up to 21× at end of window.
                    # Effect: voltage at load drops; for resistive loads current drops slightly;
                    # increased I²R dissipation in connector (heat source outside the IC).
                    t_norm = np.linspace(0, 1, n_fault)
                    # Exponential-squared aging curve — slow start, accelerating
                    k_age = fi.intensity * 20.0
                    r_connector_aged = ch.connector_r_ohm * (1.0 + k_age * t_norm ** 2)
                    r_total = ch.harness_r_ohm + r_connector_aged  # Ω
                    # Voltage at load falls with aging R (current stays roughly constant — eFuse
                    # doesn't adjust; the load sees reduced voltage)
                    voltage[sl] = bus_voltage[sl] - current[sl] * r_total + self.rng.normal(0, 0.01, n_fault)
                    # For resistive loads: I = V_load / R_load — voltage drop reduces current slightly
                    # Model this as a correction proportional to relative voltage reduction
                    v_nominal_load = bus_voltage[sl] - current[sl] * (ch.harness_r_ohm + ch.connector_r_ohm)
                    v_aged_load = voltage[sl]
                    # Avoid division by zero; correction only where v_nominal_load > 0
                    with np.errstate(invalid="ignore", divide="ignore"):
                        i_ratio = np.where(
                            v_nominal_load > 0.5,
                            np.clip(v_aged_load / v_nominal_load, 0.5, 1.0),
                            1.0,
                        )
                    current[sl] *= i_ratio
                    status[sl] = DeviceStatus.WARNING.value

                case FaultType.OPEN_LOAD:
                    # Wire break / terminal corrosion — channel commanded ON but load is open.
                    # Current drops to near-zero leakage (< 1 mA, mostly noise).
                    # State remains True (CDD keeps the gate driven) — the IC cannot
                    # distinguish open-load from a very low-current load without DIAGNOSIS.
                    # After ol_blank_time_ms, the IC's DIAGNOSIS cycle confirms open load
                    # and sets the OPEN_LOAD_DIAG status bit in the SPI register.
                    ol_threshold = ch.ol_threshold_a if ch.ol_threshold_a > 0 else ch.nominal_current_a * 0.03
                    current[sl] = self.rng.normal(0.0005, 0.0002, n_fault)
                    # State stays ON (gate commanded) — IC cannot self-detect without DIAGNOSIS
                    state[sl] = True
                    # Leakage stays well below threshold — leave trip/overload clear
                    status[sl] = DeviceStatus.WARNING.value
                    # DIAGNOSIS blank time: OL flag appears after ol_blank_time_ms
                    blank_samples = max(int(ch.ol_blank_time_ms / 1000 / interval_s), 1)
                    diag_start = start_idx + blank_samples
                    if diag_start < end_idx:
                        protection_event[diag_start:end_idx] = ProtectionEvent.OPEN_LOAD_DIAG.value

                case FaultType.JUMP_START:
                    # External jump-start / booster connected: bus rises to 16–24 V.
                    # eFuse ICs have internal over-voltage clamps (typ. 28–36 V); below
                    # that the load sees elevated voltage — current scales ∝ V/R for
                    # resistive loads.  Some ICs set an over-voltage status bit.
                    # Model: linear ramp up within first 5 % of window, hold at elevated
                    # level, then ramp back over last 10 % (charger disconnected).
                    v_jump = 13.5 + fi.intensity * 10.5  # 16–24 V depending on intensity
                    envelope = self._fault_envelope_rise_fall(
                        n_fault, rise_frac=0.05, fall_frac=0.10
                    )
                    bus_voltage[sl] = 13.5 + (v_jump - 13.5) * envelope
                    # Voltage at load follows bus (negligible harness drop at low current)
                    voltage[sl] = bus_voltage[sl] + self.rng.normal(0, 0.05, n_fault)
                    # Resistive load current scales with voltage ratio
                    v_ratio = bus_voltage[sl] / 13.5
                    current[sl] = ch.nominal_current_a * v_ratio + self._composite_noise(n_fault, ch)
                    # IC reports over-voltage event once bus exceeds 16 V
                    ov_mask = bus_voltage[sl] > 16.0
                    protection_event[start_idx:end_idx][ov_mask] = ProtectionEvent.OVER_VOLTAGE.value
                    status[sl] = DeviceStatus.WARNING.value

                case FaultType.LOAD_DUMP:
                    # ISO 16750-2 load dump: alternator field collapses when a large
                    # battery/load is suddenly disconnected.  Bus spikes to ~40 V for
                    # 50–400 ms then decays exponentially back to ~13.5 V.
                    # Model: fast rise (1 % of window) to 40 V, exponential decay τ ≈ 15 %
                    # of window back to nominal.
                    v_peak = 40.0 * fi.intensity  # ~28–40 V
                    t_norm = np.linspace(0, 1, n_fault)
                    # Fast spike then exponential decay
                    tau_norm = 0.15  # decay time constant as fraction of window
                    spike = v_peak * np.exp(-t_norm / tau_norm)
                    bus_voltage[sl] = 13.5 + spike + self.rng.normal(0, 0.2, n_fault)
                    voltage[sl] = bus_voltage[sl] + self.rng.normal(0, 0.1, n_fault)
                    # IC over-voltage clamp fires — current briefly spikes then IC shuts off
                    peak_samples = max(int(n_fault * 0.05), 1)
                    current[start_idx : start_idx + peak_samples] *= 1.5  # brief inrush
                    current[start_idx + peak_samples : end_idx] = self.rng.normal(
                        0.001, 0.0005, n_fault - peak_samples
                    )  # IC off during clamp
                    state[start_idx + peak_samples : end_idx] = False
                    trip[start_idx : end_idx] = True
                    protection_event[start_idx:end_idx] = ProtectionEvent.OVER_VOLTAGE.value
                    status[sl] = DeviceStatus.FAULT.value

                case FaultType.COLD_CRANK:
                    # Cold-crank battery sag: starter motor draws 200–600 A, bus collapses
                    # to 7–9 V for 3–5 s then recovers as engine fires.
                    # ISO 16750-2 profile: 30 ms pre-crank at nominal, drop to 6 V, recover
                    # linearly over crank window.  Model uses a U-shape.
                    v_sag = 13.5 - fi.intensity * 6.5  # 7–13.5 V
                    # U-shape: ramp down in first 10 %, hold at sag, ramp up over last 20 %
                    ramp_down = max(int(n_fault * 0.10), 1)
                    hold_end = max(int(n_fault * 0.80), ramp_down + 1)
                    ramp_up_n = n_fault - hold_end
                    crank_v = np.empty(n_fault)
                    crank_v[:ramp_down] = np.linspace(13.5, v_sag, ramp_down)
                    crank_v[ramp_down:hold_end] = v_sag
                    if ramp_up_n > 0:
                        crank_v[hold_end:] = np.linspace(v_sag, 13.5, ramp_up_n)
                    bus_voltage[sl] = crank_v + self.rng.normal(0, 0.15, n_fault)
                    voltage[sl] = bus_voltage[sl] + self.rng.normal(0, 0.08, n_fault)
                    # Resistive loads see reduced current proportional to voltage sag
                    v_ratio_crank = np.clip(bus_voltage[sl] / 13.5, 0.3, 1.2)
                    current[sl] = ch.nominal_current_a * v_ratio_crank + self._composite_noise(n_fault, ch)
                    # Under-voltage warning when bus < 9 V
                    uv_mask = bus_voltage[sl] < 9.0
                    status[start_idx:end_idx] = np.where(
                        uv_mask, DeviceStatus.FAULT.value, DeviceStatus.WARNING.value
                    )

            fault_active[sl] = fi.fault_type.value
            severity[sl] = fi.intensity

        # --- Compute temperature from RC thermal model using actual current ---
        temperature = self._rc_thermal(np.nan_to_num(current, nan=0.0), ch, interval_s)
        # Add small measurement noise
        temperature += self.rng.normal(0, 0.2, n)
        temperature = np.clip(temperature, -40, ch.thermal_shutdown_c + 10)

        # --- Thermal shutdown protection ---
        # If junction temp exceeds the IC's thermal shutdown threshold,
        # the IC turns off the channel until temperature drops below
        # a hysteresis point (typ. 20°C below shutdown).
        thermal_limit = ch.thermal_shutdown_c
        thermal_hysteresis = 20.0  # typical IC hysteresis
        thermally_off = False
        for i in range(n):
            if not thermally_off and temperature[i] >= thermal_limit:
                thermally_off = True
            elif thermally_off and temperature[i] < (thermal_limit - thermal_hysteresis):
                thermally_off = False

            if thermally_off:
                current[i] = self.rng.normal(0.001, 0.0005)
                state[i] = False
                trip[i] = True
                protection_event[i] = ProtectionEvent.THERMAL_SHUTDOWN.value
                status[i] = DeviceStatus.FAULT.value

        # Update status for thermal warnings
        for fi in faults:
            if fi.fault_type == FaultType.THERMAL_DRIFT:
                start_idx = int(fi.start_s / interval_s)
                end_idx = min(start_idx + int(fi.duration_s / interval_s), n)
                sl = slice(start_idx, end_idx)
                status[sl] = np.where(temperature[sl] > 80, DeviceStatus.WARNING.value, status[sl])

        # --- ISENSE sensing-chain gain error ---
        # Apply k_ILIS temperature drift + R_ILIS tolerance to convert true
        # load current into the value that the CDD ADC actually measures.
        # Thermal-shutdown zeros (near zero, not NaN) and NaN dropped packets
        # both propagate correctly through the multiplicative chain.
        current = self._apply_isense_chain(current, temperature, ch)

        # ADC quantization on final current signal
        if ch.current_adc_bits < 16:
            adc_range = ch.max_current_a * 1.5
            lsb = adc_range / (2**ch.current_adc_bits)
            valid_mask = ~np.isnan(current)
            current[valid_mask] = np.round(current[valid_mask] / lsb) * lsb

        # ADC quantization on voltage signal (separate, typically lower resolution)
        if ch.voltage_adc_bits < 16:
            v_adc_range = ch.nominal_voltage_v * 3.0  # full-scale ~40V for 13.5V nominal
            v_lsb = v_adc_range / (2**ch.voltage_adc_bits)
            v_valid = ~np.isnan(voltage)
            voltage[v_valid] = np.round(voltage[v_valid] / v_lsb) * v_lsb

        # --- Build DataFrame directly (vectorized — avoids per-row loop) ---
        # Compute cumulative resets vectorized
        trip_edges = np.diff(trip.astype(int), prepend=0)
        trip_rising = (trip_edges == 1).astype(int)
        cum_resets_from_edges = np.cumsum(trip_rising)
        cum_resets = np.maximum(cum_resets_from_edges, reset_counter.astype(int))

        # Replace NaN with None for nullable columns
        current_out = current.copy()
        voltage_out = voltage.copy()

        ch_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "channel_id": ch.channel_id,
                "current_a": current_out,
                "voltage_v": voltage_out,
                "temperature_c": temperature,
                "state_on_off": state,
                "trip_flag": trip,
                "overload_flag": overload,
                "protection_event": protection_event,
                "reset_counter": cum_resets,
                "pwm_duty_pct": pwm,
                "device_status": status,
            }
        )

        # Build labels for fault windows
        fault_mask = fault_active != FaultType.NONE.value
        if fault_mask.any():
            label_df = pd.DataFrame(
                {
                    "timestamp": np.array(timestamps)[fault_mask],
                    "channel_id": ch.channel_id,
                    "fault_type": fault_active[fault_mask],
                    "severity": severity[fault_mask],
                    "description": [f"{ft} on {ch.channel_id}" for ft in fault_active[fault_mask]],
                }
            )
        else:
            label_df = pd.DataFrame(
                columns=["timestamp", "channel_id", "fault_type", "severity", "description"]
            )

        return ch_df, label_df
