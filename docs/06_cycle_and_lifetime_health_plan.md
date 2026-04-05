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

### B. PNDS-Style Ring Buffer

Persistent, fixed-size buffer.

Store completed cycle summaries, not raw telemetry.

Default rule:

- one compact summary per completed cycle per system or zone
- channel-level detail only for abnormal channels

This is the core automotive tradeoff.

### C. Compact Lifetime State

Persistent scalar state, not rich history.

Examples:

- current baseline EWMA
- temperature baseline EWMA
- thermal burden score
- trip burden score
- retry burden score
- anomaly burden score
- last health score
- trend direction

## How PNDS Should Be Used

Treat PNDS as the retention and upload layer for cycle summaries.

Recommended flow:

1. Streaming logic runs during the open cycle.
2. The active accumulator collects counters.
3. At cycle close, VIP computes one compact summary record.
4. That record is written to the PNDS ring buffer.
5. Upload occurs when the buffer is full, after a time threshold such as 28 days, or earlier for critical cases.

### Recommended PNDS Record

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

## How Load Spectra Should Be Used

Load spectra are useful, but only selectively.

They should not mean full histograms for every eFuse on the vehicle.

Use them for:

- critical channels
- high-duty channels
- channels already in degraded state
- top-N worst channels in each system

Keep them coarse, for example 8 to 16 bins.

Useful spectra types:

- current amplitude
- temperature dwell
- trip count per cycle
- retry count per cycle
- voltage sag depth or dwell
- thermal swing magnitude

For most channels, simple proxies are enough:

- EWMA current
- EWMA temperature
- time-over-threshold counters
- trip and retry counts
- decayed burden score

## How PNDS And Load Spectra Fit Together

- PNDS stores cycle summaries
- spectra represent long-term exposure for selected channels

Recommended pattern:

1. During a cycle, update counters and temporary exposure accumulators.
2. At cycle close, generate one PNDS summary.
3. Update compact lifetime state.
4. Update spectra only for tracked channels.
5. Let backend/cloud perform richer multi-cycle analysis.

## Health Scoring

### Cycle Health

Use a small set of stable metrics:

- anomaly count
- trip count
- retry count
- peak current relative to nominal
- max temperature relative to threshold
- high-load dwell
- high-temperature dwell
- dominant fault severity

Conceptually:

$$
CycleStress = w_1 E_{anomaly} + w_2 E_{trip} + w_3 E_{retry} + w_4 B_{thermal} + w_5 B_{current}
$$

Then map to:

- cycle stress score
- cycle health band

### Lifetime Health

Combine:

- cumulative burden
- recent trend
- repeated protection behavior
- deviation from baseline

Edge-side lifetime state stays compact.

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

### On Edge

- cycle accumulator in RAM
- PNDS-like ring buffer with compact system-level cycle summaries
- channel detail only for abnormal channels
- scalar lifetime state for top-risk channels

### In Backend

- cycle summary ingestion
- lifetime score reconstruction
- trend views
- top-risk ranking

### Minimum Outputs

- cycle stress score
- cycle health band
- lifetime health score
- lifetime trend direction
- maintenance priority

## Phased Implementation

### Phase 1

- implement cycle tracking
- implement cycle-close summary generation
- persist compact PNDS-style records

### Phase 2

- add compact lifetime state on edge
- compute health score and trend from cycle summaries

### Phase 3

- add selective load spectra for critical or degraded channels
- add backend lifetime reconstruction and dashboard views

### Phase 4

- add DTC correlation and service-facing interpretation

## Final Position

Do not build this as full per-eFuse lifetime history on vehicle.

Build it as:

- compact cycle summaries
- exception-based channel detail
- small persistent lifetime state on edge
- richer lifetime analysis off vehicle

That is the design most likely to survive real automotive constraints.
