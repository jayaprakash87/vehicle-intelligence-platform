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
        fine_times = np.arange(n_finest) * finest_s  # seconds from t0

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
                    bus_voltage = np.pad(bus_voltage, (0, n_samples - len(bus_voltage)), mode="edge")

            ch_faults = [f for f in self.cfg.fault_injections if f.channel_id == ch.channel_id]
            ch_df, label_df = self._generate_channel(timestamps, ch, ch_faults, bus_voltage, interval_s)
            all_telem.append(ch_df)
            all_labels.append(label_df)

        telem_df = pd.concat(all_telem, ignore_index=True) if all_telem else pd.DataFrame()
        labels_df = pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame()

        log.info(
            "Generated %d telemetry rows, %d labels across %d channels",
            len(telem_df), len(labels_df), len(self.cfg.channels),
        )
        return telem_df, labels_df

    # ------------------------------------------------------------------
    # Bus voltage generation (shared across channels)
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
            # Normalize to ~0.1A std
            pink *= 0.1 / (np.std(pink) + 1e-12)
            noise += pink

        # ADC quantization noise
        adc_range = ch.max_current_a * 1.5  # full-scale range
        lsb = adc_range / (2 ** ch.adc_bits)
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
        P = I² · R_ds(on)
        """
        n = len(current)
        temp = np.empty(n)
        t_j = ch.t_ambient_c  # initial junction temp = ambient
        alpha = interval_s / ch.tau_thermal_s  # discrete time constant

        for i in range(n):
            power_w = current[i] ** 2 * ch.r_ds_on_ohm
            t_steady = ch.t_ambient_c + power_w * ch.r_thermal_kw
            # Exponential approach to steady state
            t_j = t_j + alpha * (t_steady - t_j)
            temp[i] = t_j

        return np.clip(temp, -40, 150)

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
            else:
                return current  # resistive: no transient
        else:
            n_inrush = max(int(ch.inrush_duration_ms / 1000 / interval_s), 1)
            factor = ch.inrush_factor

        n_inrush = min(n_inrush, len(current))
        # Exponential decay from peak to nominal
        decay = np.exp(-3 * np.arange(n_inrush) / n_inrush)  # ~95% settled by end
        inrush_envelope = 1.0 + (factor - 1.0) * decay
        current[:n_inrush] *= inrush_envelope

        return current

    # ------------------------------------------------------------------
    # Fault waveform shaping
    # ------------------------------------------------------------------

    def _fault_envelope_rise_fall(self, n_fault: int, rise_frac: float = 0.15, fall_frac: float = 0.15) -> np.ndarray:
        """Generate a trapezoidal envelope with exponential rise and fall."""
        env = np.ones(n_fault)
        n_rise = max(int(n_fault * rise_frac), 1)
        n_fall = max(int(n_fault * fall_frac), 1)
        # Exponential rise
        env[:n_rise] = 1 - np.exp(-3 * np.arange(n_rise) / n_rise)
        # Exponential fall
        env[-n_fall:] = np.exp(-3 * np.arange(n_fall) / n_fall)
        return env

    def _oscillating_envelope(self, n_fault: int, freq_cycles: float = 3.0, damping: float = 2.0) -> np.ndarray:
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
        reset_counter: np.ndarray,
        ch: ChannelMeta,
        fault_start: int,
        fault_end: int,
        interval_s: float,
    ) -> None:
        """Model eFuse trip → channel off → cooldown → retry cycle.

        Mutates current, state, trip, reset_counter arrays in-place.
        """
        cooldown_samples = max(int(ch.cooldown_s / interval_s), 1)
        retries_done = 0
        i = fault_start

        # First trip: detect overcurrent
        trip_point = fault_start + max(int(0.003 / interval_s), 1)  # ~3ms trip time
        if trip_point >= fault_end:
            return

        while i < fault_end and retries_done <= ch.max_retries:
            # Trip event
            trip[i:fault_end] = True
            state[i:fault_end] = False
            # Channel off during cooldown — current drops to leakage
            cooldown_end = min(i + cooldown_samples, fault_end)
            current[i:cooldown_end] = self.rng.normal(0.001, 0.0005, cooldown_end - i)
            state[i:cooldown_end] = False

            retries_done += 1
            reset_counter[cooldown_end:fault_end] = retries_done

            # After cooldown, retry (current comes back, may re-trip)
            i = cooldown_end
            if i < fault_end:
                state[i:fault_end] = True
                trip[i:fault_end] = False
                # If still in fault window, current will be high again → next trip
                # Advance to next trip point
                next_trip = min(i + max(int(0.010 / interval_s), 1), fault_end)
                i = next_trip

        # If max retries exhausted, latch off for remainder
        if retries_done > ch.max_retries and i < fault_end:
            current[i:fault_end] = self.rng.normal(0.0, 0.0005, fault_end - i)
            state[i:fault_end] = False
            trip[i:fault_end] = True

    # ------------------------------------------------------------------
    # Per-channel generation
    # ------------------------------------------------------------------

    def _generate_channel(
        self,
        timestamps: list[datetime],
        ch: ChannelMeta,
        faults: list[FaultInjection],
        bus_voltage: np.ndarray,
        interval_s: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        n = len(timestamps)

        # --- Nominal baselines ---
        # Current: nominal + composite noise
        current = np.full(n, ch.nominal_current_a) + self._composite_noise(n, ch)

        # Apply load turn-on transient
        current = self._apply_inrush(current, ch, interval_s)

        # Voltage: derived from shared bus - resistive drop through harness
        harness_r = 0.020  # 20 mΩ typical harness resistance
        voltage = bus_voltage - current * harness_r + self.rng.normal(0, 0.01, n)

        state = np.ones(n, dtype=bool)
        trip = np.zeros(n, dtype=bool)
        overload = np.zeros(n, dtype=bool)
        reset_counter = np.zeros(n, dtype=int)
        pwm = np.full(n, 100.0)
        status = np.full(n, DeviceStatus.OK.value, dtype=object)
        fault_active = np.full(n, FaultType.NONE.value, dtype=object)
        severity = np.zeros(n)

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
                    envelope = self._fault_envelope_rise_fall(n_fault, rise_frac=0.05, fall_frac=0.0)
                    peak = ch.max_current_a * (1.0 + fi.intensity * 0.5)
                    current[sl] = peak * envelope + self._composite_noise(n_fault, ch) * 0.5
                    trip[sl] = True
                    overload[sl] = True
                    status[sl] = DeviceStatus.FAULT.value
                    # Protection response: trip → off → cooldown → retry
                    self._apply_protection(current, state, trip, reset_counter, ch, start_idx, end_idx, interval_s)

                case FaultType.INTERMITTENT_OVERLOAD:
                    # Damped oscillation — repeated overcurrent bursts
                    osc = self._oscillating_envelope(n_fault, freq_cycles=4.0 * fi.intensity, damping=1.5)
                    current[sl] = ch.nominal_current_a + osc * (ch.max_current_a * 1.2 - ch.nominal_current_a) * fi.intensity
                    current[sl] += self._composite_noise(n_fault, ch) * 0.3
                    overload_mask = current[sl] > ch.fuse_rating_a * 0.9
                    overload[start_idx:end_idx] = overload_mask
                    status[sl] = np.where(overload_mask, DeviceStatus.WARNING.value, DeviceStatus.OK.value)

                case FaultType.VOLTAGE_SAG:
                    # Exponential voltage drop (like battery under heavy load)
                    envelope = self._fault_envelope_rise_fall(n_fault, rise_frac=0.10, fall_frac=0.20)
                    sag_depth = ch.nominal_voltage_v * 0.3 * fi.intensity
                    voltage[sl] = bus_voltage[sl] - sag_depth * envelope + self.rng.normal(0, 0.05, n_fault)
                    status[sl] = DeviceStatus.WARNING.value

                case FaultType.THERMAL_DRIFT:
                    # Gradually increasing current draw (simulating insulation breakdown)
                    ramp = np.linspace(1.0, 1.0 + 0.3 * fi.intensity, n_fault)
                    current[sl] *= ramp
                    # Temperature effect is handled by the RC thermal model below
                    status[sl] = DeviceStatus.WARNING.value

                case FaultType.NOISY_SENSOR:
                    # Burst noise with varying amplitude — not flat Gaussian
                    burst_envelope = self._oscillating_envelope(n_fault, freq_cycles=6.0, damping=0.5)
                    noise_scale = 2.0 * fi.intensity
                    current[sl] += burst_envelope * self.rng.normal(0, noise_scale, n_fault)
                    voltage[sl] += burst_envelope * self.rng.normal(0, noise_scale * 0.2, n_fault)

                case FaultType.DROPPED_PACKET:
                    drop_mask = self.rng.random(n_fault) < 0.4 * fi.intensity
                    current[start_idx:end_idx] = np.where(drop_mask, np.nan, current[sl])
                    voltage[start_idx:end_idx] = np.where(drop_mask, np.nan, voltage[sl])

                case FaultType.GRADUAL_DEGRADATION:
                    # Slow exponential ramp (aging contact resistance)
                    t_norm = np.linspace(0, 1, n_fault)
                    deg = 1.0 + fi.intensity * 0.5 * (np.exp(2 * t_norm) - 1) / (np.e**2 - 1)
                    current[sl] *= deg
                    pwm[sl] = np.clip(100 - np.linspace(0, 30 * fi.intensity, n_fault), 0, 100)
                    status[sl] = DeviceStatus.WARNING.value

            fault_active[sl] = fi.fault_type.value
            severity[sl] = fi.intensity

        # --- Compute temperature from RC thermal model using actual current ---
        temperature = self._rc_thermal(np.nan_to_num(current, nan=0.0), ch, interval_s)
        # Add small measurement noise
        temperature += self.rng.normal(0, 0.2, n)
        temperature = np.clip(temperature, -40, 150)

        # Update status for thermal warnings
        for fi in faults:
            if fi.fault_type == FaultType.THERMAL_DRIFT:
                start_idx = int(fi.start_s / interval_s)
                end_idx = min(start_idx + int(fi.duration_s / interval_s), n)
                sl = slice(start_idx, end_idx)
                status[sl] = np.where(temperature[sl] > 80, DeviceStatus.WARNING.value, status[sl])

        # ADC quantization on final current signal
        if ch.adc_bits < 16:
            adc_range = ch.max_current_a * 1.5
            lsb = adc_range / (2 ** ch.adc_bits)
            valid_mask = ~np.isnan(current)
            current[valid_mask] = np.round(current[valid_mask] / lsb) * lsb

        # --- Build DataFrame directly (vectorized — avoids per-row loop) ---
        # Compute cumulative resets vectorized
        trip_edges = np.diff(trip.astype(int), prepend=0)
        trip_rising = (trip_edges == 1).astype(int)
        cum_resets_from_edges = np.cumsum(trip_rising)
        cum_resets = np.maximum(cum_resets_from_edges, reset_counter.astype(int))

        # Replace NaN with None for nullable columns
        current_out = current.copy()
        voltage_out = voltage.copy()

        ch_df = pd.DataFrame({
            "timestamp": timestamps,
            "channel_id": ch.channel_id,
            "current_a": current_out,
            "voltage_v": voltage_out,
            "temperature_c": temperature,
            "state_on_off": state,
            "trip_flag": trip,
            "overload_flag": overload,
            "reset_counter": cum_resets,
            "pwm_duty_pct": pwm,
            "device_status": status,
        })

        # Build labels for fault windows
        fault_mask = fault_active != FaultType.NONE.value
        if fault_mask.any():
            label_df = pd.DataFrame({
                "timestamp": np.array(timestamps)[fault_mask],
                "channel_id": ch.channel_id,
                "fault_type": fault_active[fault_mask],
                "severity": severity[fault_mask],
                "description": [f"{ft} on {ch.channel_id}" for ft in fault_active[fault_mask]],
            })
        else:
            label_df = pd.DataFrame(columns=["timestamp", "channel_id", "fault_type", "severity", "description"])

        return ch_df, label_df
