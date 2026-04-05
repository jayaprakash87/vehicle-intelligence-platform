# Product & Positioning

This is the single reference for what VIP is, why it exists, whom it serves, and how to talk about it. 

---

## 1. The One-Liner

VIP turns eFuse telemetry into fault intelligence — finding what existing diagnostics miss and adding context to what they already catch.

---

## 2. The Problem

Zone Controller ECUs read eFuse registers via SPI every 10–100 ms. The raw signals — load current, supply voltage, junction temperature, protection status — ride the CAN bus. In a typical vehicle there are 65+ channels across 4 zone controllers.

| # | Gap | Consequence |
|---|-----|-------------|
| 1 | **Data collected, not analysed.** No systematic anomaly detection runs on eFuse telemetry in production. | Gradual degradation and voltage-sag precursors go undetected. |
| 2 | **Diagnostics are reactive.** DTCs fire only on hard failures — overcurrent trip, thermal shutdown, latch-off. | Soft faults cascade into field failures. Warranty cost accrues. |
| 3 | **No reusable intelligence layer.** Every OEM writes per-ECU, per-platform diagnostic routines from scratch. | 6–12 months of engineering per vehicle line, non-transferable. |
| 4 | **Test-bench ≠ production.** Algorithms tuned on XCP at 10 ms break on CAN at 100 ms. | Two code-bases, two validation cycles. |
| 5 | **Protection events are invisible.** The IC distinguishes SCP, I²t, thermal shutdown, and latch-off — but that information never reaches the diagnostic layer. | Root-cause analysis is impossible without manual probing. |

### Who feels this

| Stakeholder | Pain |
|---|---|
| **OEM body/chassis engineers** | Writing one-off diagnostics per platform (6–12 months, non-reusable) |
| **Tier-1 eFuse suppliers** (Infineon, ST, NXP) | Customers ignore the diagnostic data their ICs provide |
| **Fleet operators** (EV fleets, commercial) | Unplanned downtime from electrical faults |
| **Warranty / quality teams** | Failures surface as warranty claims months later |

---

## 3. What VIP Is

An edge-deployable, config-driven software pipeline that ingests eFuse telemetry, extracts rolling features, detects anomalies, classifies fault types with root-cause explanations, and emits alerts — all on the vehicle.

### What VIP is not

- Not a CDD or AUTOSAR integration — runs as application software on Linux.
- Not a replacement for eFuse hardware protection — ICs still trip on overcurrent; VIP adds intelligence on top.
- Not another copy of the OEM's existing DTC routines.
- Not cloud-only — edge-first, cloud aggregation comes later.

---

## 4. How VIP Relates to Existing Diagnostics

This is the most important positioning point.

**VIP should not be sold as:** "We raise DTCs too."

**VIP should be sold as:** "We find what existing DTCs miss, and we add context to what existing DTCs already catch."

Concretely:

1. **Existing OEM DTCs stay in place.** If a hard failure already has a production DTC, VIP does not create a second one.
2. **VIP adds earlier and richer detection** — gradual degradation, intermittent overload patterns, thermal drift, voltage-sag precursors, repeated stress before a hard failure.
3. **VIP enriches existing diagnostics** — when a hard fault does occur, VIP provides earlier warning that it was developing, protection-mechanism context, cross-channel context, and severity information.

The role of VIP is usually one of these:
- predictive alerting
- diagnostic enrichment
- new diagnostics for gaps that current DTCs do not cover

---

## 5. How to Explain VIP

**To someone non-technical:**

> Existing diagnostics usually detect hard electrical failures after the problem is already severe. VIP continuously analyzes smart eFuse telemetry to catch earlier warning patterns, classify the likely fault type, and help engineers understand root cause before the issue becomes a confirmed vehicle failure.

**To engineers:**

> VIP is a reusable fault-intelligence layer on top of eFuse telemetry. It performs short-window streaming detection, rules-based fault classification, and protection-aware analysis. It is not intended to blindly duplicate existing DTC logic; it detects precursor conditions, enriches hard-fault diagnostics, and supports later correlation into OEM diagnostic frameworks.

**To product or management:**

> OEMs already have DTCs for hard failures. VIP catches earlier patterns that typical threshold diagnostics miss and turns raw eFuse data into actionable insights reusable across programs.

---

## 6. What Exists Today

| Area | Status |
|------|--------|
| Synthetic telemetry generation | Done |
| Feature engineering (10 rolling features) | Done |
| Anomaly detection (Isolation Forest) | Done |
| Rules-based fault classification (7 types) | Done |
| Streaming alert emission with edge hardening | Done |
| 17 IC families, 65-channel topology | Done |
| Multi-rate support (CAN + XCP) | Done |
| 248 tests passing | Done |
| Cycle-level health summaries | Not yet |
| Lifetime health state | Not yet |
| DTC correlation / confirmation logic | Not yet |
| Cloud / dashboard layer | Not yet |

Current description: VIP is a predictive electrical fault-intelligence prototype with streaming alerts and fault classification, not yet a complete production diagnostic stack.

---

## 7. The Data Moat

Synthetic data gets you in the door. Measurement data is the product.

```
Stage 1: SYNTHETIC     Stage 2: MEASUREMENT     Stage 3: PRODUCTION     Stage 4: FLEET
Physics-based sim      XCP + CAN recordings     Live CAN with           Cross-vehicle
Proves pipeline        Re-trains on real         engineer-reviewed       Transfer learning
No OEM data needed     signal distributions      alerts as labels        across platforms
    v0.1                   v0.5                    v1.0–v2.0                v3.0
```

- Stage 1 eliminates the NDA blocker.
- Stage 2 is the critical inflection — real noise floors, real thermal dynamics, real protection timing.
- Stage 3 closes the feedback loop — engineers confirm or dismiss alerts; those labels train supervised classifiers.
- Stage 4 is the defensible moat — patterns transfer across OEMs because the underlying physics is shared.

---

## 8. Market & Go-to-Market

**Segment 1 — eFuse IC Suppliers** (Infineon, ST, NXP). They sell smart eFuses but struggle to demonstrate the value of the diagnostic data. VIP ships as the "intelligence layer" reference with their evaluation kits. License per OEM design-in.

**Segment 2 — OEM Body / Chassis Electronics Teams.** These teams write one-off diagnostic code. VIP replaces that with a configurable pipeline. Platform license per vehicle line + annual support.

**Segment 3 — Fleet Operators.** Retrofit VIP on a telemetry gateway. Fleet-level anomaly dashboard. SaaS per vehicle per month. Requires cloud layer (v1.0+).

### Competitive positioning

| Approach | Example | VIP's differentiation |
|---|---|---|
| Custom OEM diagnostics | Every OEM, in-house | Config-driven and reusable — YAML, not C |
| General vehicle analytics | Upstream, Sibros | Not specialised for eFuse physics |
| Cloud-only telemetry | AWS IoT FleetWise | Edge-first — runs on the vehicle |
| Research tools | MATLAB / Simulink | Deployable software with a hardened runtime |

### Defensibility

1. Domain-specific IC catalog with validated electrical parameters
2. Physics-based simulation as bootstrap — competitors need real data before they can demo
3. Continuous learning flywheel — every deployment improves models
4. Protection-event classification — nobody else exposes SCP vs. I²t vs. thermal shutdown for analytics

---

## 9. Technical Strategy

### Architecture principles

1. **Edge-first.** Runs on Jetson-class hardware (4 GB RAM, ARM). Cloud is for aggregation.
2. **Config-driven.** Vehicle topology, fault scenarios, thresholds — all YAML. New vehicle line = zero code changes.
3. **Model-agnostic.** `AnomalyDetector` interface swaps IF for LSTM, XGBoost, or transformer without pipeline changes.
4. **Protocol-transparent.** CAN at 100 ms and XCP at 10 ms flow through the same pipeline.
5. **Physics-faithful simulation.** Exercises the full dynamic range or demos and model transfer will fail.
6. **Continuous learning.** Every deployment generates data; the system closes the loop.

### Model evolution

| Phase | Model | Data gate |
|---|---|---|
| **v0.1** | Isolation Forest + rules | Unlabeled synthetic telemetry |
| **v1.0** | LSTM autoencoder | ~1 M unlabeled production rows per channel |
| **v2.0** | XGBoost / LightGBM supervised | ~10 k labeled fault windows |
| **v3.0** | Temporal transformer | Fleet-scale cross-vehicle corpus |

Transitions are data-gated, not calendar-gated.

---

## 10. Roadmap

### Phase 1 — Validate (→ v0.5)

Prove VIP detects real faults that current diagnostics miss. Partner with 1 eFuse supplier, run on real CAN/XCP data, port to Jetson, add MQTT alerts and a minimal dashboard.

Exit: VIP detects ≥ 3 fault types that the OEM's current diagnostics don't surface.

### Phase 2 — Design-In (→ v1.0)

VIP is included in 1 OEM's vehicle development program. Train LSTM on production data, add OTA model push, pass security review.

Exit: VIP running in pre-production vehicles. Alert accuracy validated by OEM diagnostic engineers.

### Phase 3 — Scale (→ v2.0)

Production deployment on 1 vehicle line. Supervised classifier from accumulated labels. Fleet dashboard. 2nd OEM evaluation.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| OEM data access blocked | Synthetic demo requires zero OEM data. Supplier eval boards as alternative. |
| Real data ≠ synthetic | Physics-based sim narrows gap. Transfer learning designed in. Small real-data retrain closes residual gap. |
| Edge hardware constraints | IF model is ~100 KB. Feature engine is pure numpy. 4 GB Jetson is the design floor. |
| OEM integration timeline too long | Self-contained replay demo. Supplier channel as faster entry. |
| Single-person team risk | 248 tests, fully documented architecture, config-driven design. |

---

## 12. Why Now

1. **Zone architecture adoption.** OEMs consolidating to 4–6 zone controllers — each hosts multiple eFuse ICs. The data density is new.
2. **Smart eFuse proliferation.** Infineon PROFET+2, STMicro VIPower, NXP MC33xxx — SPI-readable diagnostics ship in the silicon. Nobody builds the software to use it.
3. **Software-defined vehicles.** Linux-based compute platforms make running analytics on a vehicle ECU realistic, not a research project.

VIP sits at the intersection: the data exists (smart eFuses), the compute exists (Linux gateways), the buyer cares (warranty costs, regulatory pressure). No one has built the intelligence layer to connect them.
