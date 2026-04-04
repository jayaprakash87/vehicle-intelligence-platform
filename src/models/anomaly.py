"""Anomaly detection and fault classification models.

Two complementary approaches:
1. Isolation Forest — unsupervised anomaly scoring (no labels needed).
2. Rules-based classifier — maps feature patterns to known fault types
   for interpretability and early prototyping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config.models import ModelConfig
from src.schemas.telemetry import FaultType, ProtectionEvent
from src.utils.logging import get_logger

log = get_logger(__name__)

# Feature columns consumed by the anomaly model
ANOMALY_FEATURES = [
    "rolling_rms_current",
    "rolling_mean_current",
    "rolling_max_current",
    "spike_score",
    "temperature_slope",
    "trip_frequency",
    "degradation_trend",
]


class AnomalyDetector:
    """Isolation Forest wrapper for unsupervised anomaly detection."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.cfg = config or ModelConfig()
        self.model: Optional[IsolationForest] = None
        # Calibration from training data — used for stable score normalization
        self._train_score_offset: float = 0.0
        self._train_score_scale: float = 1.0

    def train(self, df: pd.DataFrame, labels_df: pd.DataFrame | None = None) -> None:
        """Fit on feature-enriched telemetry.

        If contamination is 'auto' and labels_df is provided, computes the
        true fault fraction.  Otherwise falls back to sklearn's 'auto' mode.
        """
        X = self._prepare(df)
        contamination = self.cfg.anomaly_contamination
        if contamination == "auto":
            if labels_df is not None and len(labels_df) > 0:
                fault_fraction = len(labels_df) / len(df)
                contamination = max(0.01, min(fault_fraction, 0.5))
                log.info("Auto-contamination from labels: %.3f", contamination)
            else:
                contamination = "auto"
        self.model = IsolationForest(
            n_estimators=self.cfg.anomaly_n_estimators,
            contamination=contamination,
            random_state=42,
        )
        self.model.fit(X)
        # Calibrate score normalization from training distribution
        raw = self.model.decision_function(X)
        self._train_score_offset = float(raw.min())
        self._train_score_scale = float(raw.max() - raw.min())
        if self._train_score_scale < 1e-12:
            self._train_score_scale = 1.0
        log.info("Trained IsolationForest on %d samples, %d features", *X.shape)

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Return anomaly scores normalized to 0–1 (higher = more anomalous).

        Scores are calibrated against the training distribution so they are
        comparable across batches.  Values may exceed [0, 1] if test data
        is far outside training range (clipped for safety).
        Cold-start rows (NaN features) get score 0.0.
        """
        if self.model is None:
            raise RuntimeError("Model not trained — call train() first")
        X = self._prepare(df)
        mask = self._prepare_mask
        raw = self.model.decision_function(X)
        normalized = 1 - (raw - self._train_score_offset) / self._train_score_scale
        normalized = np.clip(normalized, 0.0, 1.0)
        # Rebuild full-length series with 0.0 for cold-start rows
        result = pd.Series(0.0, index=df.index, name="anomaly_score")
        result.loc[mask[mask].index] = normalized
        return result

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean anomaly labels. Cold-start rows default to False."""
        if self.model is None:
            raise RuntimeError("Model not trained — call train() first")
        X = self._prepare(df)
        mask = self._prepare_mask
        preds = self.model.predict(X)
        result = pd.Series(False, index=df.index, name="is_anomaly")
        result.loc[mask[mask].index] = preds == -1
        return result

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path or self.cfg.model_dir) / "anomaly_detector.joblib"
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "train_score_offset": self._train_score_offset,
            "train_score_scale": self._train_score_scale,
        }
        joblib.dump(payload, p)
        log.info("Saved anomaly model to %s", p)
        return p

    def load(self, path: str | Path | None = None) -> None:
        p = Path(path or self.cfg.model_dir) / "anomaly_detector.joblib"
        payload = joblib.load(p)
        if isinstance(payload, dict):
            self.model = payload["model"]
            self._train_score_offset = payload.get("train_score_offset", 0.0)
            self._train_score_scale = payload.get("train_score_scale", 1.0)
        else:
            # Backward compatibility with old format (bare model)
            self.model = payload
        log.info("Loaded anomaly model from %s", p)

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in ANOMALY_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        subset = df[ANOMALY_FEATURES]
        # Drop rows where rolling features haven't warmed up (NaN from cold start)
        # rather than filling with 0 which would mask "unknown" as "stable"
        valid_mask = subset.notna().all(axis=1)
        self._prepare_mask = valid_mask  # stash for callers that need row alignment
        X = subset[valid_mask].values
        if len(X) == 0:
            raise ValueError(
                "No valid rows after dropping NaN features (cold-start window too large)"
            )
        return X


# ---------------------------------------------------------------------------
# Rules-based fault classifier
# ---------------------------------------------------------------------------


class RulesFaultClassifier:
    """Interpretable rules that map feature patterns to known fault types.

    This is intentionally simple — good for demos, debugging, and baseline
    comparison before training a learned classifier.
    """

    def classify(self, row: pd.Series | dict) -> tuple[FaultType, float, list[str]]:
        """Classify a single feature-enriched row.

        Returns (fault_type, confidence, likely_causes).
        Uses protection_event when available to sharpen root-cause determination.
        """
        r = row if isinstance(row, dict) else row.to_dict()

        spike = r.get("spike_score", 0)
        trip_freq = r.get("trip_frequency", 0)
        temp_slope = r.get("temperature_slope", 0)
        deg_trend = r.get("degradation_trend", 0)
        r.get("rolling_rms_current", 0)
        overload = r.get("overload_flag", False)
        trip = r.get("trip_flag", False)
        r.get("current_a", 0) or 0
        voltage = r.get("voltage_v", 0) or 0
        missing_rate = r.get("missing_rate", 0) or 0

        # Protection event context (available when generator or CDD populates it)
        pe = r.get("protection_event", ProtectionEvent.NONE.value)
        scp_count = r.get("scp_count", 0) or 0
        i2t_count = r.get("i2t_count", 0) or 0
        latch_off_count = r.get("latch_off_count", 0) or 0
        thermal_shutdown_count = r.get("thermal_shutdown_count", 0) or 0

        causes: list[str] = []

        # --- Latch-off: max retries exhausted, channel locked ---
        if pe == ProtectionEvent.LATCH_OFF.value or latch_off_count > 0:
            causes.append("eFuse exhausted max auto-retry attempts — channel latched off")
            if scp_count > i2t_count:
                causes.append("Preceding SCP trips suggest wiring short-circuit")
            elif i2t_count > 0:
                causes.append("Preceding I²t trips suggest sustained overload")
            return FaultType.OVERLOAD_SPIKE, min(0.8 + latch_off_count * 0.1, 1.0), causes

        # --- Thermal shutdown ---
        if pe == ProtectionEvent.THERMAL_SHUTDOWN.value or thermal_shutdown_count > 0:
            causes.append("Junction temperature exceeded thermal shutdown limit")
            if temp_slope > 0.2:
                causes.append("Sustained temperature rise — check cooling or load current")
            return FaultType.THERMAL_DRIFT, min(0.7 + thermal_shutdown_count * 0.1, 1.0), causes

        # Dropped packets — high missing data rate
        if missing_rate > 0.1:
            causes.append("High rate of missing signal samples indicates packet loss")
            return FaultType.DROPPED_PACKET, min(missing_rate / 0.4, 1.0), causes

        # Overload spike — high spike + trip, refined by protection event type
        if spike > 4.0 and trip:
            if pe == ProtectionEvent.SCP.value:
                causes.append("Short-circuit protection fired — check wiring or connector")
            elif pe == ProtectionEvent.I2T.value:
                causes.append("I²t energy-integral trip — sustained overcurrent, not instantaneous")
            else:
                causes.append("Sudden high current draw exceeding fuse rating")
            return FaultType.OVERLOAD_SPIKE, min(spike / 6.0, 1.0), causes

        # Intermittent overload — moderate spike + repeating trips
        if trip_freq > 2 and overload:
            causes.append("Repeated transient overcurrent events")
            if scp_count > 0 and i2t_count > 0:
                causes.append("Mixed SCP and I²t trips — intermittent short with load stress")
            elif scp_count > 0:
                causes.append("Repeated SCP trips — loose connector or chafed harness")
            return FaultType.INTERMITTENT_OVERLOAD, min(trip_freq / 5.0, 1.0), causes

        # Voltage sag
        if voltage < 11.0:
            causes.append("Supply voltage below safe threshold")
            return FaultType.VOLTAGE_SAG, min((13.5 - voltage) / 4.0, 1.0), causes

        # Thermal drift
        if temp_slope > 0.3:
            causes.append("Sustained temperature rise indicating thermal issue")
            return FaultType.THERMAL_DRIFT, min(temp_slope / 1.0, 1.0), causes

        # Gradual degradation
        if deg_trend > 0.01:
            causes.append("Slow upward trend in load current suggesting wear")
            return FaultType.GRADUAL_DEGRADATION, min(deg_trend / 0.05, 1.0), causes

        # Noisy sensor — high spike but no trip
        if spike > 2.5 and not trip and not overload:
            causes.append("High variance without protection response — possible sensor noise")
            return FaultType.NOISY_SENSOR, min(spike / 5.0, 1.0), causes

        return FaultType.NONE, 0.0, []

    def classify_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify every row, returning columns: predicted_fault, fault_confidence, likely_causes."""
        results = df.apply(lambda row: self.classify(row), axis=1, result_type="expand")
        results.columns = ["predicted_fault", "fault_confidence", "likely_causes"]
        results["predicted_fault"] = results["predicted_fault"].apply(
            lambda f: f.value if hasattr(f, "value") else str(f)
        )
        return results
