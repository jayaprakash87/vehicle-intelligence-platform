"""Integration test — full pipeline end-to-end."""

from src.config.models import default_config
from src.features.engine import FeatureEngine
from src.ingestion.normalizer import Normalizer
from src.inference.pipeline import InferencePipeline
from src.models.anomaly import AnomalyDetector
from src.models.evaluation import evaluate
from src.schemas.telemetry import FaultInjection, FaultType
from src.simulation.generator import TelemetryGenerator


def test_full_pipeline():
    cfg = default_config()
    cfg.simulation.duration_s = 20.0
    cfg.simulation.fault_injections = [
        FaultInjection(
            channel_id="ch_01",
            fault_type=FaultType.OVERLOAD_SPIKE,
            start_s=5.0,
            duration_s=3.0,
            intensity=0.8,
        ),
    ]
    cfg.features.window_size = 20
    cfg.features.min_periods = 5

    # 1. Simulate
    gen = TelemetryGenerator(cfg.simulation)
    telem_df, labels_df = gen.generate()
    assert len(telem_df) > 0

    # 2. Normalize
    norm = Normalizer()
    telem_df = norm.normalize(telem_df)

    # 3. Features + Train
    engine = FeatureEngine(cfg.features)
    feat_df = engine.compute(telem_df)

    detector = AnomalyDetector(cfg.model)
    detector.train(feat_df, labels_df=labels_df)

    # 4. Inference
    pipeline = InferencePipeline(cfg.features, cfg.model, detector)
    scored = pipeline.run_batch(telem_df)

    assert "is_anomaly" in scored.columns
    assert "predicted_fault" in scored.columns
    assert scored["is_anomaly"].any(), "Expected at least one anomaly from injected fault"

    # 5. Evaluate against ground truth
    metrics = evaluate(scored, labels_df)
    overall = metrics["overall"]
    assert overall["recall"] > 0.1, f"Recall too low: {overall['recall']}"
    assert overall["true_positives"] > 0, "Expected true positives in fault window"

    # 6. Convert to schema objects
    results = pipeline.to_inference_results(scored.head(10))
    assert len(results) == 10
