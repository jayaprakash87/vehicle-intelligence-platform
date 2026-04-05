# Metrics Reference

Complete reference for every cycle metric and lifetime load-spectrum histogram in VIP.

---

## 1. Cycle Metrics (CycleSummary)

Produced once per completed cycle by `CycleAccumulator`. Held in RAM as a list on the edge device.

### 1.1 Identity & Timing

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `cycle_id` | `str` | UUID-based unique identifier (12 hex chars) | Generated at cycle open |
| `cycle_type` | `str` | Cycle kind: `ignition`, `drive`, `charge` | Config: `EdgeConfig.cycle_type` |
| `open_timestamp` | `datetime` | UTC timestamp when cycle opened | Rising edge of `boundary_column` |
| `close_timestamp` | `datetime` | UTC timestamp when cycle closed | Falling edge of `boundary_column` |
| `duration_s` | `float` | Cycle length in seconds | $\text{close\_timestamp} - \text{open\_timestamp}$ |
| `sample_count` | `int` | Total scored rows ingested during cycle | Incremented per row in `_accumulate_row()` |

### 1.2 Event Counters

| Field | Type | Description | How Accumulated | Saturation |
|-------|------|-------------|-----------------|------------|
| `anomaly_count` | `int` | Rows where `is_anomaly == True` | `+1` per anomalous row | Unbounded |
| `trip_count` | `int` | Rows where `trip_flag == True` | `+1` per trip row | Unbounded |
| `retry_count` | `int` | Peak value of `reset_counter` column | `max(current, row.reset_counter)` | Unbounded |

### 1.3 Peak Values

| Field | Type | Unit | Description | How Accumulated |
|-------|------|------|-------------|-----------------|
| `peak_current_a` | `float` | A | Maximum instantaneous current observed | `max()` over all rows |
| `peak_temperature_c` | `float` | °C | Maximum junction temperature observed | `max()` over all rows |

### 1.4 Dwell Times

Dwell = cumulative time spent above (or below) a threshold during the cycle. Estimated as `sample_interval_s` per qualifying row.

| Field | Type | Unit | Threshold | Default | Config |
|-------|------|------|-----------|---------|--------|
| `high_load_dwell_s` | `float` | s | `current_a > high_current_threshold_a` | 80% of fuse rating | `CycleAccumulator(high_current_threshold_a=...)` |
| `high_temp_dwell_s` | `float` | s | `temperature_c > high_temp_threshold_c` | 125 °C | `CycleAccumulator(high_temp_threshold_c=...)` |
| `voltage_sag_dwell_s` | `float` | s | `voltage_v < low_voltage_threshold_v` | 10.0 V | `CycleAccumulator(low_voltage_threshold_v=...)` |

### 1.5 Dominant Fault

| Field | Type | Description | Calculation |
|-------|------|-------------|-------------|
| `dominant_fault` | `FaultType` | Most-voted fault across all rows in the cycle | `Counter.most_common(1)` over `predicted_fault` where `fault_confidence > 0` |
| `dominant_fault_confidence` | `float` | Fraction of fault votes for the dominant fault | $\frac{\text{votes for top fault}}{\text{total fault votes}}$ |

Possible `FaultType` values: `none`, `overload_spike`, `intermittent_overload`, `voltage_sag`, `thermal_drift`, `noisy_sensor`, `dropped_packet`, `gradual_degradation`.

### 1.6 Cycle Stress Score

A single composite metric summarising how stressed this cycle was.

$$
\text{CycleStress} = 0.35 \cdot E_{\text{anomaly}} + 0.25 \cdot E_{\text{trip}} + 0.15 \cdot E_{\text{retry}} + 0.25 \cdot B_{\text{thermal}}
$$

Where:

| Component | Symbol | Formula | Clamp |
|-----------|--------|---------|-------|
| Anomaly rate | $E_{\text{anomaly}}$ | $\frac{\text{anomaly\_count}}{\text{sample\_count}}$ | [0, 1] |
| Trip burden | $E_{\text{trip}}$ | $\frac{\text{trip\_count}}{10}$ | [0, 1] |
| Retry burden | $E_{\text{retry}}$ | $\frac{\text{retry\_count}}{5}$ | [0, 1] |
| Thermal burden | $B_{\text{thermal}}$ | $\frac{\text{high\_temp\_dwell\_s}}{\text{duration\_s}}$ | [0, 1] |

Final stress is clamped to [0, 1].

### 1.7 Health Band

Mapped from cycle stress:

| Band | Stress Range | Meaning |
|------|-------------|---------|
| **NOMINAL** | < 0.15 | Normal operation |
| **MONITOR** | 0.15 – 0.40 | Elevated stress, worth observing |
| **DEGRADED** | 0.40 – 0.70 | Significant stress, investigate |
| **CRITICAL** | ≥ 0.70 | Severe stress, act immediately |

### 1.8 Memory Footprint

| Component | Size | Notes |
|-----------|------|-------|
| CycleSummary (Pydantic) | ~320 bytes | In-memory Python object |
| CycleSummary (JSON) | ~400 bytes | Serialised for upload |
| Active accumulator (RAM) | ~200 bytes | Counters + peaks + dwell, reset each cycle |

---

## 2. Lifetime Load-Spectrum Histograms

Primary state of `LifetimeHealthState`. Six fixed-bin histograms, each with 8 bins (7 edges). Updated once per completed cycle by `LifetimeHealthTracker.ingest()`.

### 2.1 Histogram Structure

Each histogram has:
- **N-1 edges** defining upper bounds of the first N-1 bins
- **N bins** where the last bin captures everything ≥ the last edge
- **Record(value)**: increment the appropriate bin
- **upper_fraction(k)**: fraction of total counts in the top-k bins (used for health derivation)

### 2.2 All Six Histograms

#### peak_current_a — Peak Current Distribution

| Property | Value |
|----------|-------|
| **Source field** | `CycleSummary.peak_current_a` |
| **Unit** | Amperes (A) |
| **Edges** | `[2, 5, 8, 12, 15, 20, 30]` |
| **Health weight** | 0.20 |

| Bin | Range | Typical loads |
|-----|-------|---------------|
| 0 | < 2 A | LEDs, sensors, keep-alive |
| 1 | 2 – 5 A | Dome lights, indicators |
| 2 | 5 – 8 A | Locks, medium resistive |
| 3 | 8 – 12 A | Headlamps, fog lights |
| 4 | 12 – 15 A | Wipers, windows |
| 5 | 15 – 20 A | Seat adjust, defroster |
| 6 | 20 – 30 A | Seat heaters, HVAC blower |
| 7 | ≥ 30 A | PDU high-current, starter |

---

#### peak_temperature_c — Peak Junction Temperature Distribution

| Property | Value |
|----------|-------|
| **Source field** | `CycleSummary.peak_temperature_c` |
| **Unit** | °C |
| **Edges** | `[40, 60, 80, 100, 120, 140, 160]` |
| **Health weight** | 0.20 |

| Bin | Range | Interpretation |
|-----|-------|----------------|
| 0 | < 40 °C | Cold / idle |
| 1 | 40 – 60 °C | Normal ambient + light load |
| 2 | 60 – 80 °C | Moderate load, warm cabin |
| 3 | 80 – 100 °C | Heavy load, underhood proximity |
| 4 | 100 – 120 °C | Sustained high load |
| 5 | 120 – 140 °C | Approaching thermal limits |
| 6 | 140 – 160 °C | Near thermal shutdown threshold |
| 7 | ≥ 160 °C | Thermal shutdown territory |

---

#### cycle_stress — Cycle Stress Score Distribution

| Property | Value |
|----------|-------|
| **Source field** | `CycleSummary.cycle_stress` |
| **Unit** | Ratio [0, 1] |
| **Edges** | `[0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80]` |
| **Health weight** | 0.20 |

| Bin | Range | Health Band Overlap |
|-----|-------|---------------------|
| 0 | < 0.05 | Deep nominal |
| 1 | 0.05 – 0.10 | Nominal |
| 2 | 0.10 – 0.15 | Nominal / monitor boundary |
| 3 | 0.15 – 0.25 | Monitor |
| 4 | 0.25 – 0.40 | Monitor / degraded boundary |
| 5 | 0.40 – 0.60 | Degraded |
| 6 | 0.60 – 0.80 | Degraded / critical boundary |
| 7 | ≥ 0.80 | Critical |

---

#### trips_per_cycle — Protection Trip Count Distribution

| Property | Value |
|----------|-------|
| **Source field** | `CycleSummary.trip_count` |
| **Unit** | Count |
| **Edges** | `[1, 2, 3, 5, 8, 12, 20]` |
| **Health weight** | 0.15 |

| Bin | Range | Interpretation |
|-----|-------|----------------|
| 0 | < 1 (= 0) | No trips — clean cycle |
| 1 | 1 | Single trip — transient event |
| 2 | 2 | Repeated — possible intermittent fault |
| 3 | 3 – 4 | Frequent — investigate load/wiring |
| 4 | 5 – 7 | Persistent overcurrent pattern |
| 5 | 8 – 11 | Severe — nearing latch-off territory |
| 6 | 12 – 19 | Protection system under heavy stress |
| 7 | ≥ 20 | Extreme — likely hard fault |

---

#### retries_per_cycle — Auto-Retry Count Distribution

| Property | Value |
|----------|-------|
| **Source field** | `CycleSummary.retry_count` |
| **Unit** | Count |
| **Edges** | `[1, 2, 3, 5, 8, 12, 20]` |
| **Health weight** | 0.10 |

| Bin | Range | Interpretation |
|-----|-------|----------------|
| 0 | < 1 (= 0) | No retries — clean |
| 1 | 1 | Single retry — transient cleared |
| 2 | 2 | Repeated — borderline load |
| 3 | 3 – 4 | Multiple retries per cycle |
| 4 | 5 – 7 | High retry rate — wiring suspect |
| 5 | 8 – 11 | Getting close to max_retries exhaust |
| 6 | 12 – 19 | Very high abuse |
| 7 | ≥ 20 | Extreme — latch-off likely reached |

---

#### thermal_dwell_fraction — High-Temperature Dwell Fraction Distribution

| Property | Value |
|----------|-------|
| **Source field** | Derived: `min(CycleSummary.high_temp_dwell_s / duration_s, 1.0)` |
| **Unit** | Ratio [0, 1] |
| **Edges** | `[0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75]` |
| **Health weight** | 0.15 |

| Bin | Range | Interpretation |
|-----|-------|----------------|
| 0 | < 1% | Negligible hot time |
| 1 | 1 – 5% | Brief thermal excursion |
| 2 | 5 – 10% | Moderate exposure |
| 3 | 10 – 20% | Significant thermal load |
| 4 | 20 – 35% | Sustained high temperature |
| 5 | 35 – 50% | Heavy thermal stress |
| 6 | 50 – 75% | Majority of cycle above threshold |
| 7 | ≥ 75% | Near-continuous overtemperature |

---

## 3. Derived Lifetime Health

These values are **recomputed** from histogram shape on every cycle close — they are not primary state.

### 3.1 Health Score

$$
\text{HealthScore} = 1.0 - \sum_{i=1}^{6} w_i \cdot \text{upper\_fraction}_i(k)
$$

Where $\text{upper\_fraction}_i(k)$ is the fraction of total counts in the top-$k$ bins for histogram $i$, and $k$ defaults to 2.

| Histogram | Weight ($w_i$) |
|-----------|---------------|
| peak_current_a | 0.20 |
| peak_temperature_c | 0.20 |
| cycle_stress | 0.20 |
| trips_per_cycle | 0.15 |
| retries_per_cycle | 0.10 |
| thermal_dwell_fraction | 0.15 |

**Interpretation**: A system where counts steadily migrate from lower to upper bins will see `upper_fraction` grow and health score drop.

### 3.2 Health Band (Lifetime)

| Band | Health Score Range | Action |
|------|--------------------|--------|
| **NOMINAL** | > 0.85 | No action needed |
| **MONITOR** | 0.60 – 0.85 | Track — schedule review |
| **DEGRADED** | 0.30 – 0.60 | Investigate — early upload |
| **CRITICAL** | ≤ 0.30 | Immediate attention — priority upload |

### 3.3 Trend Direction

Computed from the last $N$ health scores (default $N = 5$, configurable via `lifetime_trend_window`).

Method: compare average of first half to average of second half.

$$
\delta = \overline{\text{score}_{\text{second half}}} - \overline{\text{score}_{\text{first half}}}
$$

| Trend | Condition | Meaning |
|-------|-----------|---------|
| **IMPROVING** | $\delta > +0.05$ | Health recovering |
| **STABLE** | $-0.03 \leq \delta \leq +0.05$ | No significant change |
| **DEGRADING** | $-0.10 < \delta < -0.03$ | Slow decline |
| **WORSENING** | $\delta \leq -0.10$ | Rapid decline — escalate |

---

## 4. Memory Summary

| Component | Items | Size per Item | Total |
|-----------|-------|---------------|-------|
| 6 histograms × 8 bins | 48 counters | 4 bytes (uint32) | **192 bytes** |
| 6 × 7 edges | 42 floats | 4 bytes (float32) | 168 bytes |
| Derived scalars (score, band, trend) | 3 | ~4 bytes each | 12 bytes |
| Bookkeeping (cycle_id, timestamp, count) | 3 | ~30 bytes | 30 bytes |
| **Total LifetimeHealthState** | | | **≈ 402 bytes** |
| CycleSummary (per cycle, in RAM until upload) | 1 | ~320 bytes | 320 bytes |
| CycleAccumulator (active, RAM only) | 1 | ~200 bytes | 200 bytes |

All state fits comfortably in a single NvM block (typical automotive NvM block: 512 – 4096 bytes).

---

## 5. Configuration Reference

### EdgeConfig fields

| Field | Default | Description |
|-------|---------|-------------|
| `cycle_tracking_enabled` | `False` | Enable cycle accumulation |
| `cycle_type` | `"ignition"` | Cycle label |
| `cycle_boundary_column` | `"state_on_off"` | Column for boundary detection (`None` = manual) |
| `lifetime_tracking_enabled` | `False` | Enable lifetime histogram updates (requires cycle tracking) |
| `lifetime_trend_window` | `5` | Recent scores for trend detection |
| `lifetime_upper_bins` | `2` | Top-k bins for upper-fraction health computation |

### CycleAccumulator thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `high_current_threshold_a` | `None` (80% fuse rating) | Current above which dwell counted |
| `high_temp_threshold_c` | `125.0` | Temperature above which dwell counted |
| `low_voltage_threshold_v` | `10.0` | Voltage below which sag dwell counted |

---

## 6. Data Flow

```
Telemetry Samples (100ms)
        │
        ▼
┌───────────────────┐
│  EdgeRuntime      │   Streaming layer: score every batch
│  InferencePipeline│   Output: scored DataFrame with is_anomaly,
│                   │   anomaly_score, predicted_fault, fault_confidence
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ CycleAccumulator  │   Cycle layer: RAM counters while cycle open
│                   │   Output on close: CycleSummary
└───────┬───────────┘
        │
        ▼
┌───────────────────────┐
│ LifetimeHealthTracker │   Lifetime layer: record into histograms,
│                       │   derive health score + trend
│  6 histograms × 8 bins│   Output: LifetimeHealthState
└───────────────────────┘
```
