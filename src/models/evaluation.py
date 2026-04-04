"""Model evaluation — compare inference results against ground-truth labels.

Uses the simulator's labels_df (with exact fault windows) to compute
per-fault-type precision, recall, F1, and time-to-detect.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def evaluate(
    scored_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    tolerance_s: float = 1.0,
) -> dict:
    """Evaluate scored inference output against ground-truth fault labels.

    Args:
        scored_df: DataFrame with columns: timestamp, channel_id, is_anomaly,
                   predicted_fault, anomaly_score.
        labels_df: DataFrame with columns: timestamp, channel_id, fault_type,
                   severity (from simulator).
        tolerance_s: Time tolerance for matching detections to label windows.

    Returns:
        Dict with overall and per-fault-type metrics.
    """
    scored = scored_df.copy()
    labels = labels_df.copy()

    # Ensure datetime types
    scored["timestamp"] = pd.to_datetime(scored["timestamp"], utc=True)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)

    # Build ground-truth fault windows from labels
    fault_windows = _build_fault_windows(labels)

    # Tag each scored row: is it within a fault window?
    scored["_ground_truth_fault"] = scored.apply(
        lambda row: _match_fault_window(row, fault_windows, tolerance_s), axis=1
    )
    scored["_is_faulty"] = scored["_ground_truth_fault"] != "none"

    # Overall metrics
    tp = ((scored["is_anomaly"]) & (scored["_is_faulty"])).sum()
    fp = ((scored["is_anomaly"]) & (~scored["_is_faulty"])).sum()
    fn = ((~scored["is_anomaly"]) & (scored["_is_faulty"])).sum()
    tn = ((~scored["is_anomaly"]) & (~scored["_is_faulty"])).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    overall = {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_rows": len(scored),
        "fault_rows": int(scored["_is_faulty"].sum()),
        "nominal_rows": int((~scored["_is_faulty"]).sum()),
    }

    # Per-fault-type metrics
    per_fault = {}
    for fw in fault_windows:
        ft = fw["fault_type"]
        ch = fw["channel_id"]

        in_window = scored[
            (scored["channel_id"] == ch)
            & (scored["timestamp"] >= fw["start"] - pd.Timedelta(seconds=tolerance_s))
            & (scored["timestamp"] <= fw["end"] + pd.Timedelta(seconds=tolerance_s))
        ]
        if len(in_window) == 0:
            continue

        detected = in_window["is_anomaly"].sum()
        total = len(in_window)
        window_recall = detected / total if total > 0 else 0.0

        # Time to detect: time from fault start to first anomaly detection
        first_detection = in_window[in_window["is_anomaly"]]["timestamp"].min()
        ttd = None
        if pd.notna(first_detection):
            ttd = (first_detection - fw["start"]).total_seconds()

        # Fault classification accuracy within window
        correct_class = in_window[in_window["predicted_fault"] == ft]
        class_accuracy = len(correct_class) / total if total > 0 else 0.0

        key = f"{ft}_{ch}"
        per_fault[key] = {
            "fault_type": ft,
            "channel_id": ch,
            "window_rows": total,
            "detections": int(detected),
            "recall": round(window_recall, 4),
            "time_to_detect_s": round(ttd, 2) if ttd is not None else None,
            "classification_accuracy": round(class_accuracy, 4),
        }

    result = {"overall": overall, "per_fault": per_fault}
    log.info(
        "Evaluation: precision=%.3f recall=%.3f F1=%.3f (%d fault windows)",
        precision,
        recall,
        f1,
        len(fault_windows),
    )
    return result


def _build_fault_windows(labels: pd.DataFrame) -> list[dict]:
    """Group contiguous label rows into fault windows."""
    windows = []
    for (ch, ft), grp in labels.groupby(["channel_id", "fault_type"]):
        if ft == "none":
            continue
        grp = grp.sort_values("timestamp")
        windows.append(
            {
                "channel_id": ch,
                "fault_type": ft,
                "start": grp["timestamp"].min(),
                "end": grp["timestamp"].max(),
            }
        )
    return windows


def _match_fault_window(
    row: pd.Series,
    fault_windows: list[dict],
    tolerance_s: float,
) -> str:
    """Return the fault type if this row falls within a fault window, else 'none'."""
    ts = row["timestamp"]
    ch = row["channel_id"]
    for fw in fault_windows:
        if ch != fw["channel_id"]:
            continue
        if (
            fw["start"] - pd.Timedelta(seconds=tolerance_s)
            <= ts
            <= fw["end"] + pd.Timedelta(seconds=tolerance_s)
        ):
            return fw["fault_type"]
    return "none"
