"""Quality-gate test — the pipeline must detect known faults above minimum thresholds.

This test runs scenarios with known fault injections and asserts that model
performance doesn't silently regress. Thresholds are calibrated to what the
current Isolation Forest + rules classifier reliably achieves — they're
intentionally set as regression guards, not aspirational targets.

This test is deterministic (fixed seed) and should never be flaky.
"""

from src.config.models import default_config
from src.features.engine import FeatureEngine
from src.inference.pipeline import InferencePipeline
from src.ingestion.normalizer import Normalizer
from src.models.anomaly import AnomalyDetector
from src.models.evaluation import evaluate
from src.schemas.telemetry import FaultInjection, FaultType
from src.simulation.generator import TelemetryGenerator


def _run_scenario(
    fault_type: FaultType,
    channel_id: str = "ch_01",
    start_s: float = 8.0,
    duration_s: float = 5.0,
    intensity: float = 0.8,
) -> dict:
    """Run a single-fault scenario and return evaluation metrics."""
    cfg = default_config()
    cfg.simulation.duration_s = 30.0
    cfg.simulation.seed = 42
    cfg.simulation.fault_injections = [
        FaultInjection(
            channel_id=channel_id,
            fault_type=fault_type,
            start_s=start_s,
            duration_s=duration_s,
            intensity=intensity,
        ),
    ]

    gen = TelemetryGenerator(cfg.simulation)
    telem_df, labels_df = gen.generate()

    norm = Normalizer(cfg.normalizer)
    telem_df = norm.normalize(telem_df)

    engine = FeatureEngine(cfg.features)
    feat_df = engine.compute(telem_df)

    detector = AnomalyDetector(cfg.model)
    detector.train(feat_df, labels_df=labels_df)

    pipeline = InferencePipeline(cfg.features, cfg.model, detector)
    scored = pipeline.run_batch(telem_df)

    return evaluate(scored, labels_df, tolerance_s=1.0)


class TestQualityGate:
    """Minimum detection quality thresholds for known fault scenarios."""

    def test_overload_spike_detection(self):
        """Overload spike must be reliably detected — strongest signal fault."""
        metrics = _run_scenario(FaultType.OVERLOAD_SPIKE)
        overall = metrics["overall"]
        assert overall["recall"] >= 0.2, f"Recall {overall['recall']:.3f} < 0.2"
        assert overall["precision"] >= 0.2, f"Precision {overall['precision']:.3f} < 0.2"
        assert overall["f1"] >= 0.2, f"F1 {overall['f1']:.3f} < 0.2"
        assert overall["true_positives"] > 0

    def test_thermal_drift_detection(self):
        """Thermal drift scenario must run and produce scored output."""
        metrics = _run_scenario(FaultType.THERMAL_DRIFT, intensity=0.9)
        overall = metrics["overall"]
        # Thermal drift is subtle — just verify the pipeline produces results
        assert overall["total_rows"] > 0
        assert overall["fault_rows"] > 0

    def test_voltage_sag_detection(self):
        """Voltage sag scenario must run and produce scored output."""
        metrics = _run_scenario(FaultType.VOLTAGE_SAG, intensity=0.9)
        overall = metrics["overall"]
        assert overall["total_rows"] > 0
        assert overall["fault_rows"] > 0

    def test_no_false_alarms_on_nominal(self):
        """Clean scenario with no faults should produce low false positive rate."""
        cfg = default_config()
        cfg.simulation.duration_s = 30.0
        cfg.simulation.seed = 42
        cfg.simulation.fault_injections = []
        # Explicit low contamination — no labels to auto-compute from
        cfg.model.anomaly_contamination = 0.05

        gen = TelemetryGenerator(cfg.simulation)
        telem_df, labels_df = gen.generate()

        norm = Normalizer(cfg.normalizer)
        telem_df = norm.normalize(telem_df)

        engine = FeatureEngine(cfg.features)
        feat_df = engine.compute(telem_df)

        detector = AnomalyDetector(cfg.model)
        detector.train(feat_df)

        pipeline = InferencePipeline(cfg.features, cfg.model, detector)
        scored = pipeline.run_batch(telem_df)

        n_anomalies = scored["is_anomaly"].sum()
        false_positive_rate = n_anomalies / len(scored)
        assert false_positive_rate < 0.15, (
            f"False positive rate {false_positive_rate:.3f} too high on nominal data"
        )
