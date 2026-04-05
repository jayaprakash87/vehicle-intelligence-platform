#!/usr/bin/env python3
"""VIP Quickstart — run a full pipeline in Python and inspect results.

Usage:
    python examples/quickstart.py

This script demonstrates:
  1. Building a 3-channel vehicle topology
  2. Injecting an overload fault
  3. Generating synthetic telemetry with physics models
  4. Normalizing, extracting features, training, and running inference
  5. Evaluating detection quality
  6. Running cycle accumulation and lifetime health tracking
"""

from pathlib import Path

from src.config.models import (
    FeatureConfig,
    ModelConfig,
    NormalizerConfig,
    SimulationConfig,
    StorageConfig,
)
from src.features.engine import FeatureEngine
from src.inference.pipeline import InferencePipeline
from src.ingestion.normalizer import Normalizer
from src.models.anomaly import AnomalyDetector
from src.models.evaluation import evaluate
from src.schemas.telemetry import ChannelMeta, FaultInjection, FaultType
from src.simulation.generator import TelemetryGenerator
from src.storage.writer import StorageWriter

# ── 1. Define channels ──────────────────────────────────────────────────────

channels = [
    ChannelMeta(channel_id="ch_01", load_name="headlamp", nominal_current_a=6.0),
    ChannelMeta(channel_id="ch_02", load_name="defroster", nominal_current_a=12.0),
    ChannelMeta(channel_id="ch_03", load_name="seat_heater", nominal_current_a=8.0),
]

# ── 2. Inject an overload fault on ch_01 ────────────────────────────────────

faults = [
    FaultInjection(
        channel_id="ch_01",
        fault_type=FaultType.OVERLOAD_SPIKE,
        start_s=5.0,
        duration_s=5.0,
        intensity=0.8,
    )
]

# ── 3. Generate synthetic telemetry ─────────────────────────────────────────

sim_cfg = SimulationConfig(
    duration_s=30,
    sample_interval_ms=100,
    channels=channels,
    fault_injections=faults,
    seed=42,
)

gen = TelemetryGenerator(sim_cfg)
telemetry_df, labels_df = gen.generate()
print(f"Generated {len(telemetry_df)} telemetry rows, {len(labels_df)} labels")

# ── 4. Normalize ────────────────────────────────────────────────────────────

norm = Normalizer(NormalizerConfig())
telemetry_df = norm.normalize(telemetry_df)

# ── 5. Feature engineering ──────────────────────────────────────────────────

feat_cfg = FeatureConfig(window_duration_s=2.0, min_duration_s=0.5)
engine = FeatureEngine(feat_cfg)
features_df = engine.compute(telemetry_df)
print(f"Computed {len(features_df.columns)} feature columns")

# ── 6. Train anomaly model ──────────────────────────────────────────────────

model_cfg = ModelConfig()
detector = AnomalyDetector(model_cfg)
detector.train(features_df)
print("Anomaly model trained")

# ── 7. Run inference ────────────────────────────────────────────────────────

pipeline = InferencePipeline(feat_cfg, model_cfg, detector)
scored_df = pipeline.run_batch(telemetry_df)

n_anomalies = scored_df["is_anomaly"].sum()
n_faults = (scored_df["predicted_fault"] != FaultType.NONE.value).sum()
print(f"Detected {n_anomalies} anomalies, {n_faults} fault events")

# ── 8. Evaluate detection quality ───────────────────────────────────────────

results = evaluate(scored_df, labels_df)
overall = results["overall"]
print(
    f"Detection metrics: precision={overall['precision']:.2f}, "
    f"recall={overall['recall']:.2f}, F1={overall['f1']:.2f}"
)

# ── 9. Save results ────────────────────────────────────────────────────────

out_dir = Path("output/quickstart")
out_dir.mkdir(parents=True, exist_ok=True)
writer = StorageWriter(StorageConfig(output_dir=str(out_dir)))
writer.write_telemetry(telemetry_df)
writer.write_scored(scored_df)
writer.write_labels(labels_df)
print(f"Results saved to {out_dir}/")

# ── 10. Show per-channel summary ───────────────────────────────────────────

print("\n╭─── Per-channel summary ───╮")
for ch_id, grp in scored_df.groupby("channel_id"):
    n = len(grp)
    anom = int(grp["is_anomaly"].sum())
    top = grp.loc[grp["predicted_fault"] != "none", "predicted_fault"]
    top_fault = top.mode().iloc[0] if len(top) > 0 else "none"
    print(f"  {ch_id:12s}  rows={n:4d}  anomalies={anom:3d}  top_fault={top_fault}")
print("╰───────────────────────────╯")

# ── 11. Cycle accumulation ──────────────────────────────────────────────────

from src.edge.cycle import CycleAccumulator

accumulator = CycleAccumulator(cycle_type="ignition")
# Feed scored data — boundary detection auto-opens/closes on state_on_off
summaries = accumulator.ingest(scored_df)
# Force-close any cycle still open at end of data
final = accumulator.close()
if final:
    summaries.append(final)

print(f"\nCycle tracking: {len(summaries)} cycle(s) detected")
for s in summaries:
    print(
        f"  {s.cycle_id}  stress={s.cycle_stress:.3f}  band={s.health_band.value:8s}"
        f"  anomalies={s.anomaly_count}  trips={s.trip_count}"
        f"  peak_I={s.peak_current_a:.1f}A  peak_T={s.peak_temperature_c:.1f}°C"
    )

# ── 12. Lifetime health tracking ───────────────────────────────────────────

from src.edge.lifetime import LifetimeHealthTracker

tracker = LifetimeHealthTracker(upper_bins=2, trend_window=5)
# Feed all completed cycle summaries into lifetime tracker
for s in summaries:
    state = tracker.ingest(s)

print(f"\nLifetime health: score={state.health_score:.3f}, band={state.health_band.value}, trend={state.trend.value}")
print(f"  cycles tracked: {state.cycles_ingested}")
print("  Load spectra (counts per bin):")
print(f"    peak_current:      {state.peak_current_hist.counts}")
print(f"    peak_temperature:  {state.peak_temperature_hist.counts}")
print(f"    cycle_stress:      {state.cycle_stress_hist.counts}")
print(f"    trips_per_cycle:   {state.trips_per_cycle_hist.counts}")
print(f"    retries_per_cycle: {state.retries_per_cycle_hist.counts}")
print(f"    thermal_dwell:     {state.thermal_dwell_frac_hist.counts}")

print("\nDone! View results in the dashboard:")
print("  vip dashboard --data output/quickstart/")
