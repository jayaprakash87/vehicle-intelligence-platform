# Cycle And Lifetime Health Plan

## Purpose

VIP already does fast streaming detection.

What is missing is a practical way to answer:

1. What happened in this cycle?
2. Is this channel or system getting worse over time?
3. What compact data should stay on vehicle versus move off vehicle?

This note focuses only on that.

## Constraints

Any realistic automotive design has to assume:

- limited ECU RAM and flash
- limited persistent write budget
- limited uplink bandwidth
- no rich per-eFuse history on vehicle by default
- no large histogram bank for every channel

So the design must be:

- compact on edge
- detailed only on exception
- richer in backend/cloud

## Architecture

Use three layers.

### 1. Streaming Layer

Purpose:

- observe the live electrical behavior of each channel
- detect short-window anomalies
- classify likely short-window fault candidates
- emit alerts when thresholds are crossed

Output:

- live channel state
- anomaly events
- protection events
- fault candidates

In the current VIP implementation, this layer already exists.

Important clarification:

An anomaly or predicted fault does not define the eFuse by itself.

At streaming level, VIP is combining three things:

1. static channel identity and electrical profile
2. live telemetry state such as current, voltage, temperature, trip, and protection behavior
3. short-window diagnosis such as anomaly score and predicted fault

So this layer tells us what the channel is doing right now and how suspicious that behavior looks.

It does not yet maintain a persistent health state for the channel across cycles or across lifetime.

### 2. Cycle Layer

Purpose:

- summarize one completed ignition, drive, or charge cycle
- turn noisy event streams into one stable session assessment

Output:

- cycle stress score
- cycle health band
- dominant fault candidate
- trip, retry, and protection burden

### 3. Lifetime Layer

Purpose:

- track whether a system is stable, degrading, or critical over many cycles

Output:

- lifetime health score
- trend direction
- maintenance priority
- hotspot ranking

## What Lives On Vehicle

### A. Active Cycle Accumulator

RAM only, while a cycle is open.

Keep only compact counters such as:

- anomaly count
- trip count
- retry count
- peak current
- peak temperature
- high-load dwell
- high-temperature dwell
- voltage-sag dwell

### B. Cycle Summary List (RAM)

Hold completed cycle summaries in memory until uploaded.

Default rule:

- one compact summary per completed cycle per system or zone
- channel-level detail only for abnormal channels
- upload to backend clears the list

Note: real PNDS lives in the Zone Controller AUTOSAR layer (NvM, C/C++).
VIP runs on a separate Linux compute module and does not write to or emulate PNDS.
On-vehicle persistence of cycle summaries is the ZC's responsibility, not VIP's.

### C. Compact Lifetime State

Six fixed-bin histogram counters (8 bins each), updated once per cycle close.

Histograms track:

- peak current distribution
- peak temperature distribution
- cycle stress distribution
- trips per cycle distribution
- retries per cycle distribution
- thermal dwell fraction distribution

Derived from histogram shape (not stored separately):

- health score (1 − weighted upper-tail fractions)
- health band
- trend direction (half-split comparison over recent scores)

Total memory: ~402 bytes. Fits a single NvM block.

See [07_metrics_reference.md](07_metrics_reference.md) for exact bin edges, weights, and formulas.

## Cycle Summary Flow

Recommended flow:

1. Streaming logic runs during the open cycle.
2. The active accumulator collects counters.
3. At cycle close, VIP computes one compact summary record.
4. The summary is held in RAM.
5. Upload to backend occurs on schedule, buffer-count threshold, or priority override for critical cases.

### Cycle Summary Record

Keep it small.

Suggested fields:

- cycle id
- cycle type
- close timestamp
- scope type: system, zone, or channel exception
- scope id
- cycle stress score
- cycle health band
- anomaly count
- trip count
- retry count
- peak current
- peak temperature
- dominant fault candidate
- compact protection summary
- upload priority

## Default Versus Exception Detail

Do not store rich channel-level detail for every eFuse by default.

### Default

Always keep:

- system-level or zone-level cycle summaries
- compact lifetime state

### Exception Mode

Keep channel-level detail only when a channel is interesting, for example:

- alert emitted
- repeated protection events
- health below threshold
- strong drift from baseline
- top-N worst channels in the cycle

## How Load Spectra Are Used

Load spectra are implemented as system-level histograms on the edge device.

Six histograms, each with 8 bins (7 edges):

- peak current amplitude (A)
- peak junction temperature (°C)
- cycle stress score (ratio)
- trip count per cycle
- retry count per cycle
- thermal dwell fraction per cycle

Each bin is a simple uint32 counter — one increment per completed cycle.

Health is derived from upper-tail fractions: the proportion of total cycles that landed in the top-k bins (default k=2). A system where counts migrate from lower to upper bins sees its health score drop.

This approach:

- preserves distribution shape (unlike EWMA)
- answers questions like "how many cycles had peak current above 20 A?"
- maps naturally to Miner’s rule / Wöhler curves for fatigue analysis
- is NvM-friendly (~192 bytes for all 48 counters)
- matches standard automotive load-spectra practice (8-bin fixed histograms)

## How Cycle Summaries And Load Spectra Fit Together

- Cycle summaries capture what happened in one session
- Load-spectrum histograms track cumulative exposure over many cycles

Implemented pattern:

1. During a cycle, CycleAccumulator updates RAM counters per scored row.
2. At cycle close, one CycleSummary is generated with stress score and health band.
3. LifetimeHealthTracker records 6 values from the summary into load-spectrum histograms.
4. Lifetime health score, band, and trend are recomputed from histogram shape.
5. Summaries stay in memory until upload to backend.

## Health Scoring

### Cycle Health

Computed at cycle close from accumulated counters:

$$
CycleStress = 0.35 \cdot E_{anomaly} + 0.25 \cdot E_{trip} + 0.15 \cdot E_{retry} + 0.25 \cdot B_{thermal}
$$

Each component is clamped to [0, 1]. See [07_metrics_reference.md](07_metrics_reference.md) for per-component formulas.

Band mapping:

| Band | Stress Range |
|------|--------------|
| NOMINAL | < 0.15 |
| MONITOR | 0.15 – 0.40 |
| DEGRADED | 0.40 – 0.70 |
| CRITICAL | ≥ 0.70 |

### Lifetime Health

Derived from histogram upper-tail fractions with weighted combination:

$$
HealthScore = 1.0 - \sum_{i=1}^{6} w_i \cdot \text{upper\_fraction}_i(k)
$$

Where $k$ is the number of top bins to consider (default 2).

Weights: peak current 0.20, peak temperature 0.20, cycle stress 0.20, trips 0.15, retries 0.10, thermal dwell 0.15.

| Band | Score Range |
|------|-------------|
| NOMINAL | > 0.85 |
| MONITOR | 0.60 – 0.85 |
| DEGRADED | 0.30 – 0.60 |
| CRITICAL | ≤ 0.30 |

Trend is detected by comparing the average of the first and second halves of a sliding window of recent health scores.

Backend-side analytics can add:

- peer comparison
- fleet percentile ranking
- long-term degradation ranking

## Cycle Definition

Cycle boundaries should be configurable.

Typical options:

- ignition cycle
- drive cycle
- charge cycle
- wake-sleep session

For the first implementation, support one or two cycle types only.

## Upload Policy

Use the 28-day or buffer-full rule, but add priority override:

1. normal summaries: scheduled upload or buffer-full
2. degraded systems: earlier upload
3. critical events: immediate or next-connectivity upload

## What Users Should See

### Recent / Edge View

- current top stressed systems
- last cycle health
- currently degraded channels

### Backend / Dashboard View

- lifetime health trend
- worst systems over 30 or 90 days
- recurring cycle patterns
- systems entering monitor, degraded, or critical bands
- fleet percentile ranking for comparable systems

## MVP

### On Edge (Implemented)

- cycle accumulator in RAM (CycleAccumulator)
- in-memory cycle summary list with upload to backend
- channel detail only for abnormal channels
- histogram-based lifetime state (6 × 8-bin load spectra, ~402 bytes)
- health score, band, and trend derived from histogram shape

### In Backend (Future)

- cycle summary ingestion
- lifetime score reconstruction
- trend views
- top-risk ranking

### Minimum Outputs (Implemented)

- cycle stress score
- cycle health band
- lifetime health score
- lifetime trend direction
- per-histogram distribution data

## Phased Implementation

### Phase 1 — Done

- ✅ cycle tracking (CycleAccumulator)
- ✅ cycle-close summary generation (CycleSummary, HealthBand)
- ✅ summaries held in RAM, available for upload

### Phase 2 — Done

- ✅ histogram-based lifetime state on edge (6 histograms × 8 bins)
- ✅ health score derived from upper-tail fractions
- ✅ trend detection (half-split comparison)
- ✅ wired into EdgeRuntime (cycle close → lifetime ingest)

### Phase 3 — Next

- backend cycle summary ingestion
- backend lifetime reconstruction and dashboard views
- selective channel-level detail for degraded channels

### Phase 4

- DTC correlation and service-facing interpretation
- fleet-level comparison and percentile ranking

## Final Position

Do not build this as full per-eFuse lifetime history on vehicle.

Build it as:

- compact cycle summaries (CycleSummary, ~320 bytes each)
- exception-based channel detail
- histogram-based load spectra on edge (~402 bytes total)
- richer lifetime analysis off vehicle

That is the design most likely to survive real automotive constraints.
