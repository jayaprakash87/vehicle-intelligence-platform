"""Inference pipeline — orchestrates feature computation + model scoring.

Can be used in batch (full DataFrame) or streaming (row-by-row / mini-batch)
mode.  The edge runtime wraps this pipeline.
"""

from __future__ import annotations

import pandas as pd

from src.config.models import FeatureConfig, ModelConfig
from src.features.engine import FeatureEngine
from src.models.anomaly import AnomalyDetector, RulesFaultClassifier
from src.schemas.telemetry import FaultType, InferenceResult
from src.utils.logging import get_logger

log = get_logger(__name__)


class InferencePipeline:
    """End-to-end scoring: raw telemetry → features → anomaly score + fault class."""

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        model_config: ModelConfig | None = None,
        anomaly_model: AnomalyDetector | None = None,
    ) -> None:
        self.feature_engine = FeatureEngine(feature_config)
        self.anomaly_detector = anomaly_model or AnomalyDetector(model_config)
        self.classifier = RulesFaultClassifier()

    def run_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a full DataFrame. Returns the df with inference columns appended."""
        # 1. Compute features
        feat_df = self.feature_engine.compute(df)

        # 2. Anomaly scoring
        if self.anomaly_detector.model is not None:
            feat_df["anomaly_score"] = self.anomaly_detector.score(feat_df)
            feat_df["is_anomaly"] = self.anomaly_detector.predict(feat_df)
        else:
            log.warning("No trained anomaly model — using spike_score as proxy")
            feat_df["anomaly_score"] = feat_df["spike_score"].clip(0, 1)
            feat_df["is_anomaly"] = feat_df["anomaly_score"] > 0.5

        # 3. Fault classification (rules-based)
        fault_results = self.classifier.classify_df(feat_df)
        feat_df["predicted_fault"] = fault_results["predicted_fault"]
        feat_df["fault_confidence"] = fault_results["fault_confidence"]
        feat_df["likely_causes"] = fault_results["likely_causes"]

        log.info(
            "Inference complete: %d rows, %d anomalies detected",
            len(feat_df),
            feat_df["is_anomaly"].sum(),
        )
        return feat_df

    def run_streaming(self, buffer: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
        """Append new_rows to buffer, recompute tail features, and score.

        Returns the scored new_rows only (not the full buffer).
        For edge use: maintain a rolling buffer externally.
        """
        combined = pd.concat([buffer, new_rows], ignore_index=True)
        scored = self.run_batch(combined)
        # Return only the new portion
        return scored.tail(len(new_rows)).reset_index(drop=True)

    def to_inference_results(self, df: pd.DataFrame) -> list[InferenceResult]:
        """Convert scored DataFrame rows to InferenceResult schema objects."""
        results = []
        for _, row in df.iterrows():
            fault_str = row.get("predicted_fault", "none")
            try:
                fault = FaultType(fault_str)
            except ValueError:
                fault = FaultType.NONE

            action = _recommend_action(fault, row.get("anomaly_score", 0))

            results.append(
                InferenceResult(
                    timestamp=row["timestamp"],
                    channel_id=row["channel_id"],
                    is_anomaly=bool(row.get("is_anomaly", False)),
                    anomaly_score=float(row.get("anomaly_score", 0)),
                    predicted_fault=fault,
                    fault_confidence=float(row.get("fault_confidence", 0)),
                    likely_causes=row.get("likely_causes", []),
                    recommended_action=action,
                )
            )
        return results


def _recommend_action(fault: FaultType, score: float) -> str:
    """Simple action lookup based on fault type."""
    actions = {
        FaultType.NONE: "No action required",
        FaultType.OVERLOAD_SPIKE: "Check load wiring; verify fuse rating matches load spec",
        FaultType.INTERMITTENT_OVERLOAD: "Inspect connector for intermittent contact; monitor load",
        FaultType.VOLTAGE_SAG: "Check battery health and alternator output",
        FaultType.THERMAL_DRIFT: "Verify heatsink mounting; check ambient airflow",
        FaultType.NOISY_SENSOR: "Inspect sensor wiring for EMI; check ADC calibration",
        FaultType.DROPPED_PACKET: "Check CAN bus termination and wiring integrity",
        FaultType.GRADUAL_DEGRADATION: "Schedule preventive maintenance; trend indicates aging load",
        FaultType.OPEN_LOAD: "Inspect wiring harness and connectors for break or corrosion; check terminal crimp integrity",
        FaultType.JUMP_START: "Verify external jump-start voltage; check eFuse OV clamp rating and surge absorber health",
        FaultType.LOAD_DUMP: "Inspect alternator suppression clamp (TVS/varistor); check eFuse over-voltage protection; review ISO 16750-2 compliance",
        FaultType.COLD_CRANK: "Check battery state-of-health and cold-cranking amps (CCA); inspect battery terminals for corrosion",
        FaultType.CONNECTOR_AGING: "Inspect harness connectors and crimp terminals for fretting wear or oxidation; re-terminate or replace corroded pins",
        FaultType.THERMAL_COUPLING: "Identify high-current co-die neighbour channel; verify per-channel de-rating meets shared-die thermal budget (check IC application note)",
        FaultType.WAKE_TRANSIENT: "Check eFuse inrush rating vs load capacitance; add pre-charge resistor or stagger KL15 turn-on sequence to limit simultaneous wake inrush",
    }
    return actions.get(fault, "Investigate further")
