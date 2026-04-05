# Product Strategy — Vehicle Intelligence Platform

## 1. The Company

**VIP** builds software that turns eFuse telemetry into fault intelligence for automotive electrical systems.

Every modern vehicle routes power through 50–100+ electronic fuses. Each eFuse IC already measures current, voltage, temperature, and protection status — the data exists on the wire. Nobody is analysing it. VIP does.

We run on the vehicle (edge-first), score every measurement, and alert when something is going wrong — before the driver notices, before the warranty claim, before the recall. We bootstrap with physics-based synthetic data so customers can demo without sharing OEM data, then progressively retrain on real measurements so the system gets smarter with every deployment.

---

## 2. The Problem

### What's happening in vehicles

Zone Controller ECUs read eFuse registers via SPI every 10–100 ms as part of the CDD (Complex Device Driver) cycle. The raw signals — load current, supply voltage, junction temperature, protection status — ride the CAN bus. In a typical vehicle there are 65+ channels across 4 zone controllers.

### What's broken

| # | Gap | Consequence |
|---|-----|-------------|
| 1 | **Data collected, not analysed.** CDD reads registers and forwards signals, but no systematic anomaly detection or fault classification runs on this data in production. | Gradual degradation, intermittent overloads, and voltage-sag precursors go undetected. |
| 2 | **Diagnostics are reactive.** DTCs fire only on hard failures — overcurrent trip, thermal shutdown, latch-off. | Soft faults cascade into field failures. Warranty cost accrues. |
| 3 | **No reusable intelligence layer.** Every OEM writes per-ECU, per-platform diagnostic routines from scratch. | 6–12 months of engineering per vehicle line, non-transferable. |
| 4 | **Test-bench ≠ production.** XCP at 10 ms on the bench, CAN at 100 ms in the car. Algorithms tuned on one rate break on the other. | Two code-bases, two validation cycles, inconsistent coverage. |
| 5 | **Protection events are invisible.** The eFuse IC distinguishes SCP (short-circuit), I²t overcurrent, thermal shutdown, and latch-off — but that information never reaches the diagnostic layer. | All trips look the same; root-cause analysis is impossible without manual probing. |

### Who feels this

| Stakeholder | Pain | Cost |
|---|---|---|
| **OEM body/chassis engineers** | Writing one-off diagnostics per platform | 6–12 months of non-reusable effort |
| **Tier-1 eFuse suppliers** (Infineon, ST, NXP) | Customers ignore the diagnostic data their ICs provide | Limits the value prop of smart eFuses; harder to justify premium ICs |
| **Fleet operators** (EV fleets, commercial) | Unplanned downtime from electrical faults | Revenue loss, stranded vehicles, SLA penalties |
| **Warranty / quality teams** | Failures surface as warranty claims months later | Reactive, expensive, reputation damage |

---

## 3. The Vision

**Make every eFuse channel in every vehicle self-diagnosing.**

A vehicle running VIP on its gateway ECU continuously scores all channels, classifies emerging faults by type and protection mechanism, and reports actionable alerts — before the fault becomes a failure. Every vehicle that runs VIP generates data that makes the next deployment smarter.

---

## 4. The Product

### What VIP is

An **edge-deployable, config-driven software pipeline** that:

1. **Ingests** eFuse telemetry from CAN (production) or XCP DAQ (test bench)
2. **Normalises** multi-rate signals onto a common analysis frame
3. **Extracts** 10 rolling signal features per channel (RMS, spike score, thermal slope, trip frequency, recovery time, degradation trend, …)
4. **Detects** anomalies using Isolation Forest (unsupervised — works from day one, no labels required)
5. **Classifies** 7 fault types with interpretable threshold rules — root cause, confidence, recommended action
6. **Tags** protection events (SCP, I²t, thermal shutdown, latch-off) so the classifier knows *which* mechanism fired, not just that a trip occurred
7. **Re-trains** on real measurement data as it accumulates — models improve continuously
8. **Runs hardened** on edge hardware: alert throttling, heartbeat monitoring, model hot-reload, graceful shutdown, run-ID isolation

### What VIP is not

- Not a CDD or AUTOSAR integration — runs as application software on Linux
- Not a replacement for eFuse protection — ICs still handle overcurrent shutdown; VIP provides intelligence on top
- Not cloud-only — edge-first, cloud aggregation is a later addition
- Not vendor-locked — works with any IC that exposes current / voltage / temperature / status

### What exists today (v0.1)

| Area | Detail |
|------|--------|
| **Codebase** | 3 600+ LOC production, 2 600+ LOC tests, 21 commits, 6 YAML configs |
| **Simulation** | Physics-based: RC thermal, composite noise (pink + quantization + EMI), load-specific inrush (motor, PTC, capacitive), F(i,t) + SCP protection cycling, bus-voltage sag |
| **eFuse catalog** | 17 IC families (Infineon + STMicroelectronics, 2A–100A + CUSTOM) with real IC part numbers and validated electrical/thermal parameters |
| **Vehicle topology** | 65-channel example across 4 zones, declarative factory, extensible to any vehicle line |
| **Telemetry schema** | `TelemetryRecord` with current, voltage, temperature, trip_flag, overload_flag, `protection_event` (SCP / I²t / thermal_shutdown / latch_off), reset_counter, PWM duty, device_status |
| **Multi-rate** | CAN (50–100 ms) + XCP dual-raster (10 ms + 50 ms), per-channel `sample_interval_ms`, auto-scaling feature windows |
| **Feature engine** | 10 derived features with time-based rolling windows |
| **Anomaly detection** | Isolation Forest (~100 KB model, sub-ms inference) |
| **Fault classification** | 7 types, rules-based, confidence scores, root-cause explanations |
| **Edge runtime** | Alert cooldown, heartbeat, signal-safe shutdown, model hot-reload, error resilience |
| **CLI** | `simulate`, `train`, `infer`, `edge`, `pipeline` — JSON logging, run-ID isolation |
| **Tests** | 184 passing across 17 test files |

---

## 5. The Data Moat

Synthetic data gets you in the door. Measurement data is the product.

### Four-stage data maturity lifecycle

```
Stage 1              Stage 2                Stage 3                 Stage 4
SYNTHETIC            MEASUREMENT            PRODUCTION              FLEET
─────────────────    ──────────────────     ───────────────────     ─────────────────────
Physics-based sim    XCP test-bench +       Live CAN from           Cross-vehicle
                     CAN recordings         production vehicles     aggregated data +
Proves pipeline      from real vehicles     with engineer-          engineer-reviewed
works                                       reviewed alerts         labels from all OEMs
                     Re-trains models
No OEM data          on real signal         Supervised classifiers  Transfer learning
required             distributions          become viable           across platforms

   v0.1                 v0.5                   v1.0–v2.0               v3.0
```

**Why this sequence matters:**

- **Stage 1** eliminates the NDA blocker. Demo in a day.
- **Stage 2** is the critical inflection.Real noise floors, real thermal dynamics, real protection timing that physics approximations can only estimate. Models are retrained, not merely evaluated.
- **Stage 3** closes the feedback loop. Engineers confirm or dismiss alerts; those labels train supervised classifiers. Every false-positive dismissed is training data.
- **Stage 4** is the defensible moat. Patterns from one OEM's headlamp circuit transfer to another's because the underlying physics (eFuse behaviour, load characteristics) are shared. More partners → better models → harder to replicate.

---

## 6. Market & Go-to-Market

### Three customer segments, in adoption order

#### Segment 1 — eFuse IC Suppliers (Infineon, STMicroelectronics, NXP)

**Why first:** They sell smart eFuses but struggle to demonstrate the value of the diagnostic data. VIP becomes the "intelligence layer" reference that ships with their evaluation kits.

**Entry:** Partner with one supplier's application engineering team. Integrate VIP on their eval board. They distribute to OEM customers as part of design-in support.

**Revenue model:** License per OEM design-in, or bundled into supplier's support package.

**Proof point needed:** Run VIP on supplier's eval board, detect faults their default demo doesn't surface.

#### Segment 2 — OEM Body / Chassis Electronics Teams

**Why next:** These teams write the one-off diagnostic code. VIP replaces that effort with a configurable, reusable pipeline.

**Entry:** POC on one vehicle line using real CAN/XCP data. Run on their existing gateway ECU or a dedicated compute module. Show detection of faults current diagnostics miss.

**Revenue model:** Platform license per vehicle line + annual support + retraining services.

**Proof point needed:** Detect ≥ 3 fault types that the OEM's current DTC routines don't surface (gradual degradation, intermittent overloads, voltage-sag precursors).

#### Segment 3 — Fleet Operators (EV fleets, commercial vehicles)

**Why later:** Requires the cloud aggregation layer and dashboard (v1.0+ infrastructure).

**Entry:** Retrofit VIP on a telemetry gateway. Provide fleet-level anomaly dashboard and predictive maintenance alerts.

**Revenue model:** SaaS per vehicle per month.

### Competitive positioning

| Approach | Example | VIP's differentiation |
|---|---|---|
| Custom OEM diagnostics | Every OEM, in-house | VIP is config-driven and reusable across platforms — YAML, not C |
| General vehicle analytics | Upstream, Sibros | Too broad; not specialised for eFuse / power-distribution physics |
| Cloud-only telemetry | AWS IoT FleetWise | Edge-first — runs on the vehicle, no cloud dependency |
| Research tools | MATLAB / Simulink | VIP is deployable software with a hardened runtime, not a simulation environment |

**Defensibility:**
1. **Domain-specific IC catalog** with validated electrical parameters — hard to replicate without the eFuse expertise
2. **Physics-based simulation** as bootstrap — competitors need real data before they can even demo
3. **Continuous learning flywheel** — every deployment generates data that improves the models; each new OEM partner makes the next one easier
4. **Protection-event classification** — nobody else exposes SCP vs. I²t vs. thermal shutdown for analytics

---

## 7. Technical Strategy

### Architecture principles

1. **Edge-first.** Everything must run on Jetson-class hardware (4 GB RAM, ARM). Cloud is for aggregation, not computation.
2. **Config-driven.** Vehicle topology, fault scenarios, feature windows, edge thresholds — all YAML. New vehicle line = zero code changes.
3. **Model-agnostic.** The `AnomalyDetector` interface (`train`, `score`, `predict`) swaps Isolation Forest for LSTM, XGBoost, or transformer without pipeline changes.
4. **Protocol-transparent.** CAN at 100 ms and XCP at 10 ms flow through the same pipeline. Feature windows auto-scale. Resampling is optional.
5. **Physics-faithful simulation.** Synthetic data must exercise the full dynamic range — RC thermal, realistic noise, protection cycling — or demos won't be credible and model transfer will fail.
6. **Continuous learning.** Every deployment generates data; the system closes the loop via measurement → retrain → deploy → feedback.

### Model evolution

| Phase | Model | Data gate |
|---|---|---|
| **v0.1** | Isolation Forest + rules | Unlabeled telemetry (synthetic) |
| **v1.0** | LSTM autoencoder | ~1 M unlabeled production rows per channel |
| **v2.0** | XGBoost / LightGBM supervised | ~10 k labeled fault windows from engineer-reviewed alerts |
| **v3.0** | Temporal transformer | Fleet-scale cross-vehicle corpus |

Transitions are data-gated, not calendar-gated. The interfaces are designed so model upgrades are a config change.

### Infrastructure evolution

| Component | v0.1 (now) | v0.5 | v1.0 | v2.0 |
|---|---|---|---|---|
| Compute | Laptop | Jetson Nano/Xavier | Production gateway ECU | Multi-ECU fleet |
| Data source | Synthetic | Real CAN/XCP via replay | Live CAN bus | Live CAN + OTA |
| Alerts | Local JSON | MQTT to cloud broker | Fleet management API | OEM service network |
| Model delivery | Local joblib | Manual deploy | OTA model push | A/B rollout |
| Monitoring | CLI + heartbeat.json | Prometheus | Grafana dashboard | Fleet SLA monitoring |
| Logging | JSON to stdout | Log aggregation (ELK) | Centralised w/ run-ID | OpenTelemetry |

---

## 8. Roadmap

### Phase 1 — Validate (months 0–6, → v0.5)

**Objective:** Prove VIP detects real faults that current diagnostics miss.

| Action | Deliverable |
|---|---|
| Partner with 1 eFuse supplier's app engineering team | Signed evaluation agreement |
| Run VIP on real CAN/XCP data from 1 vehicle platform | Detection report: VIP vs. existing DTCs |
| Port edge runtime to Jetson | Benchmark: latency, memory, 65-channel throughput |
| MQTT alert publishing | Alerts flow from edge to cloud broker |
| Minimal monitoring dashboard (Grafana/Streamlit) | Real-time channel health view |

**Exit criteria:** VIP detects ≥ 3 fault types that the OEM's current diagnostics don't surface.

### Phase 2 — Design-In (months 6–12, → v1.0)

**Objective:** VIP is included in 1 OEM's vehicle development program.

| Action | Deliverable |
|---|---|
| Adapt topology config for partner OEM's vehicle line | Custom YAML with OEM's zone/channel layout |
| Train LSTM autoencoder on 3+ months of production data | Baseline model on real signal distributions |
| OTA model push | Model versioning + edge update mechanism |
| Security review (AUTOSAR SecOC, data handling) | Compliance report |
| Pricing model finalised | License structure document |

**Exit criteria:** VIP running in pre-production vehicles. Model retrained on real data. Alert accuracy validated by OEM diagnostic engineers.

### Phase 3 — Scale (months 12–24, → v2.0)

**Objective:** Production deployment on 1 vehicle line. Pipeline validated for 2nd OEM.

| Action | Deliverable |
|---|---|
| Production release on gateway ECU | v1.0 with AUTOSAR integration guide |
| Fleet monitoring dashboard | Multi-vehicle alert triage, model performance tracking |
| Supervised classifier using accumulated labels | XGBoost/LightGBM with per-fault precision metrics |
| 2nd OEM evaluation | Adapted topology + POC results |
| RUL research spike | Feasibility report on remaining-useful-life prediction |

---

## 9. Success Metrics

### Technical KPIs

| Metric | v0.5 target | v1.0 target |
|---|---|---|
| Fault detection recall (known types) | ≥ 85 % | ≥ 92 % |
| False positive rate | ≤ 10 % | ≤ 5 % |
| Inference latency (52 ch batch) | ≤ 50 ms on Jetson | ≤ 20 ms |
| Edge memory footprint | ≤ 512 MB RSS | ≤ 256 MB RSS |
| Model artifact size | ≤ 1 MB (IF) | ≤ 5 MB (LSTM) |
| Alert-to-detection latency | ≤ 5 s | ≤ 2 s |

### Business KPIs

| Metric | 12 months | 24 months |
|---|---|---|
| OEM partners in evaluation | 2 | 4 |
| Vehicle lines in production | — | 1 |
| eFuse supplier partnerships | 1 | 2 |
| Labeled fault database | 1 k windows | 10 k windows |
| ARR | — | First license revenue |

---

## 10. Risks & Mitigations

| Risk | L | I | Mitigation |
|---|---|---|---|
| **OEM data access blocked** (NDA, security) | High | High | Synthetic demo requires zero OEM data. Air-gapped deployment. Supplier evaluation boards as an alternative starting point. |
| **Real data distribution ≠ synthetic** | Med | High | Physics-based sim narrows the gap vs. naive noise. Transfer learning designed in. Small real-data retrain closes residual gap. |
| **Edge hardware constraints** | Low | Med | IF model is ~100 KB. Feature engine is pure numpy. 4 GB Jetson Nano is the design floor. |
| **OEM integration timeline too long** | High | Med | Self-contained replay demo (no integration). Supplier channel as faster entry. |
| **IC interface fragmentation** | Med | Low | VIP operates on abstracted telemetry (current / voltage / temp / protection_event), not raw SPI registers. Transport layer adapts per vendor. |
| **Competitor enters space** | Low | Med | IC catalog + physics sim + protection-event layer = 12–18 month knowledge moat. Continuous-learning flywheel widens the gap with every deployment. |
| **Single-person team risk** | High | High | Comprehensive test suite (184 tests, 17 files). Fully documented architecture. Config-driven design minimises bus factor. |

---

## 11. Why Now

Three trends converge:

1. **Zone architecture adoption.** OEMs are consolidating from 80+ ECUs to 4–6 zone controllers. Every zone controller hosts multiple eFuse ICs. The data density is new.

2. **Smart eFuse proliferation.** Infineon PROFET+2, STMicro VIPower, NXP MC33xxx — all ship with SPI-readable diagnostics (current sense, voltage monitoring, protection status registers). The silicon already gathers the data; no one builds the software to use it.

3. **Software-defined vehicles.** The industry is moving to Linux-based compute platforms (AAOS, QNX adaptive, Stellantis STLA Brain). For the first time, running Python/C++ analytics on a vehicle ECU is a realistic deployment model, not a research project.

VIP is positioned at the intersection: the data now exists (smart eFuses), the compute now exists (Linux gateways), and the buyer now cares (warranty costs, regulatory pressure, competitive differentiation through predictive diagnostics). No one has built the intelligence layer to connect them.
