"""Edge runtime — lightweight loop that reads telemetry, scores, and emits alerts.

Designed to run on a laptop or Jetson.  Maintains a rolling buffer,
periodically computes features, runs inference, and writes alerts + summaries.

Phase D hardening: signal handling, exception resilience, health metrics,
alert rate-limiting, heartbeat, model hot-reload.
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config.models import EdgeConfig
from src.edge.cycle import CycleAccumulator
from src.edge.lifetime import LifetimeHealthTracker
from src.inference.dtc import DTCDebouncer
from src.inference.pipeline import InferencePipeline
from src.storage.writer import StorageWriter
from src.transport.alert_sinks import AlertSinkBase
from src.transport.mock_can import TransportBase
from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Health metrics
# ---------------------------------------------------------------------------


@dataclass
class IterationStats:
    """Metrics for a single loop iteration."""

    iteration: int = 0
    rows_ingested: int = 0
    inference_ms: float = 0.0
    alerts_emitted: int = 0
    buffer_rows: int = 0
    memory_rss_mb: float = 0.0


@dataclass
class RuntimeStats:
    """Cumulative metrics for the entire runtime session."""

    total_iterations: int = 0
    total_rows: int = 0
    total_alerts: int = 0
    total_errors: int = 0
    total_inference_ms: float = 0.0
    history: list[IterationStats] = field(default_factory=list)

    @property
    def avg_inference_ms(self) -> float:
        return self.total_inference_ms / max(self.total_iterations, 1)

    @property
    def rows_per_second(self) -> float:
        total_s = self.total_inference_ms / 1000.0
        return self.total_rows / max(total_s, 0.001)


# ---------------------------------------------------------------------------
# Edge runtime
# ---------------------------------------------------------------------------


class EdgeRuntime:
    """Hardened edge runtime: buffer → features → inference → alerts."""

    def __init__(
        self,
        transport: TransportBase,
        pipeline: InferencePipeline,
        edge_config: EdgeConfig | None = None,
        writer: StorageWriter | None = None,
        alert_sinks: list[AlertSinkBase] | None = None,
    ) -> None:
        self.transport = transport
        self.pipeline = pipeline
        self.cfg = edge_config or EdgeConfig()
        self.writer = writer
        self.alert_sinks: list[AlertSinkBase] = alert_sinks or []

        self._buffer: pd.DataFrame = pd.DataFrame()
        self._scored_accumulator: list[pd.DataFrame] = []
        self._alerts: list[dict] = []
        self._running = False
        self._flush_counter = 0
        self._flush_part = 0

        # Cycle tracking
        self._cycle_accumulator: CycleAccumulator | None = None
        if self.cfg.cycle_tracking_enabled:
            self._cycle_accumulator = CycleAccumulator(
                cycle_type=self.cfg.cycle_type,
                boundary_column=self.cfg.cycle_boundary_column,
            )
        self.cycle_summaries: list = []  # CycleSummary objects from completed cycles

        # Lifetime health tracking
        self._lifetime_tracker: LifetimeHealthTracker | None = None
        if self.cfg.lifetime_tracking_enabled and self.cfg.cycle_tracking_enabled:
            self._lifetime_tracker = LifetimeHealthTracker(
                trend_window=self.cfg.lifetime_trend_window,
                upper_bins=self.cfg.lifetime_upper_bins,
            )

        # Phase D state
        self.stats = RuntimeStats()
        self._consecutive_errors = 0
        self._alert_cooldown: dict[str, float] = {}  # "channel_id|fault" → last_emit_ts
        self._model_mtime: float | None = None
        self._original_sigint = None
        self._original_sigterm = None

        # DTC debounce / healing state machine
        self._dtc: DTCDebouncer | None = None
        if self.cfg.dtc_enabled:
            self._dtc = DTCDebouncer(
                fail_threshold=self.cfg.dtc_fail_threshold,
                heal_threshold=self.cfg.dtc_heal_threshold,
            )

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Pre-flight checks before entering the main loop."""
        cfg = self.cfg
        if cfg.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {cfg.batch_size}")
        if not 0 <= cfg.alert_anomaly_threshold <= 1:
            raise ValueError(
                "alert_anomaly_threshold must be between 0 and 1, "
                f"got {cfg.alert_anomaly_threshold}"
            )
        if cfg.flush_interval < 1:
            raise ValueError(f"flush_interval must be >= 1, got {cfg.flush_interval}")
        if cfg.max_consecutive_errors < 1:
            raise ValueError(
                f"max_consecutive_errors must be >= 1, got {cfg.max_consecutive_errors}"
            )
        if cfg.alert_cooldown_s < 0:
            raise ValueError(f"alert_cooldown_s must be >= 0, got {cfg.alert_cooldown_s}")
        log.info(
            "Config validated: batch=%d, flush=%d, max_err=%d",
            cfg.batch_size,
            cfg.flush_interval,
            cfg.max_consecutive_errors,
        )

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        try:
            self._original_sigint = signal.getsignal(signal.SIGINT)
            self._original_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except (OSError, ValueError):
            # Can't install signal handlers (e.g. not on main thread)
            log.debug("Could not install signal handlers (not main thread?)")

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        try:
            if self._original_sigint is not None:
                signal.signal(signal.SIGINT, self._original_sigint)
            if self._original_sigterm is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm)
        except (OSError, ValueError):
            pass

    def _handle_signal(self, signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        log.warning("Received %s — initiating graceful shutdown", sig_name)
        self._running = False

    # ------------------------------------------------------------------
    # Alert rate-limiting
    # ------------------------------------------------------------------

    def _should_emit_alert(self, channel_id: str, fault: str) -> bool:
        """Return True if this channel+fault hasn't been alerted within the cooldown."""
        if self.cfg.alert_cooldown_s <= 0:
            return True
        key = f"{channel_id}|{fault}"
        now = time.monotonic()
        last = self._alert_cooldown.get(key, 0.0)
        if now - last < self.cfg.alert_cooldown_s:
            return False
        self._alert_cooldown[key] = now
        return True

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def _write_heartbeat(self, iteration: int) -> None:
        """Write a heartbeat JSON file so external monitors can check liveness."""
        if not self.writer:
            return
        hb_path = Path(self.writer.cfg.output_dir) / "heartbeat.json"
        payload = {
            "alive": True,
            "iteration": iteration,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "total_rows": self.stats.total_rows,
            "total_alerts": self.stats.total_alerts,
            "total_errors": self.stats.total_errors,
            "avg_inference_ms": round(self.stats.avg_inference_ms, 2),
        }
        try:
            with open(hb_path, "w") as f:
                json.dump(payload, f, indent=2)
        except OSError as exc:
            log.warning("Failed to write heartbeat: %s", exc)

    # ------------------------------------------------------------------
    # Model hot-reload
    # ------------------------------------------------------------------

    def _check_model_reload(self) -> None:
        """If the model file has been updated on disk, reload it."""
        if not self.cfg.model_hot_reload:
            return
        detector = self.pipeline.anomaly_detector
        model_path = Path(detector.cfg.model_dir) / "anomaly_detector.joblib"
        if not model_path.exists():
            return
        try:
            mtime = model_path.stat().st_mtime
        except OSError:
            return
        if self._model_mtime is None:
            self._model_mtime = mtime
            return
        if mtime > self._model_mtime:
            log.info("Model file changed on disk — hot-reloading %s", model_path)
            try:
                detector.load()
                self._model_mtime = mtime
            except Exception as exc:
                log.error("Hot-reload failed, keeping previous model: %s", exc)

    # ------------------------------------------------------------------
    # Health metrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_rss_mb() -> float:
        """Return current process RSS in MB (best-effort)."""
        try:
            # Unix: read /proc/self/status or use resource module
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # macOS reports bytes, Linux reports KB
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, max_iterations: int | None = None) -> list[dict]:
        """Main loop: pull batches → score → emit.

        Args:
            max_iterations: Stop after N batches (None = run until transport exhausted).

        Returns:
            All alerts emitted during the run.
        """
        self._validate_config()
        self._install_signal_handlers()
        self._running = True
        iteration = 0
        log.info("Edge runtime started (batch_size=%d)", self.cfg.batch_size)

        try:
            while self._running:
                # --- Fetch batch ---
                batch = self.transport.batch(self.cfg.batch_size)
                if not batch:
                    log.info("Transport exhausted — stopping")
                    break

                iter_start = time.monotonic()

                try:
                    batch_df = pd.DataFrame(batch)
                    self._buffer = pd.concat([self._buffer, batch_df], ignore_index=True)

                    # Keep buffer bounded
                    max_buf = self.cfg.batch_size * 4
                    if len(self._buffer) > max_buf:
                        self._buffer = self._buffer.tail(max_buf).reset_index(drop=True)

                    # Score the new batch
                    scored = self.pipeline.run_streaming(
                        self._buffer.iloc[: -len(batch_df)],
                        batch_df,
                    )

                    # Accumulate scored data — flush periodically
                    self._scored_accumulator.append(scored)
                    self._flush_counter += 1
                    if self._flush_counter >= self.cfg.flush_interval:
                        self._flush_scored()

                    # Feed cycle accumulator
                    if self._cycle_accumulator is not None:
                        new_sums = self._cycle_accumulator.ingest(scored)
                        if new_sums:
                            self.cycle_summaries.extend(new_sums)
                            if self._lifetime_tracker is not None:
                                for cs in new_sums:
                                    self._lifetime_tracker.ingest(cs)

                    # Emit alerts only for anomalous rows above the configured threshold.
                    iter_alerts = 0
                    anomaly_mask = scored.get(
                        "is_anomaly", pd.Series(False, index=scored.index, dtype=bool)
                    )
                    score_series = scored.get(
                        "anomaly_score", pd.Series(0.0, index=scored.index, dtype=float)
                    )
                    anomalies = scored[
                        anomaly_mask.fillna(False).astype(bool)
                        & (score_series.fillna(0.0) >= self.cfg.alert_anomaly_threshold)
                    ]

                    # Update DTC state for every evaluated row (both anomalous and clean)
                    # so that the healing counter advances correctly on passing rows.
                    if self._dtc is not None:
                        from src.schemas.telemetry import FaultType as _FT

                        # Advance fail counters for anomalous rows
                        for _, row in anomalies.iterrows():
                            ch = str(row.get("channel_id", ""))
                            fault_str = row.get("predicted_fault", "none")
                            try:
                                ft = _FT(fault_str)
                            except ValueError:
                                ft = _FT.NONE
                            if ft != _FT.NONE:
                                self._dtc.update(ch, ft, fault_present=True)

                        # Advance heal counters for clean rows (not in anomaly set)
                        clean = scored.drop(index=anomalies.index, errors="ignore")
                        for _, row in clean.iterrows():
                            ch = str(row.get("channel_id", ""))
                            fault_str = row.get("predicted_fault", "none")
                            try:
                                ft = _FT(fault_str)
                            except ValueError:
                                ft = _FT.NONE
                            self._dtc.update(ch, ft, fault_present=False)

                    for _, row in anomalies.iterrows():
                        alert = self._make_alert(row)
                        # DTC gate: only publish CONFIRMED faults
                        if self._dtc is not None:
                            from src.schemas.telemetry import FaultType as _FT

                            try:
                                ft = _FT(alert["fault"])
                            except ValueError:
                                ft = _FT.NONE
                            if not self._dtc.is_confirmed(alert["channel_id"], ft):
                                continue  # PENDING — suppress, do not emit
                        if self._should_emit_alert(alert["channel_id"], alert["fault"]):
                            self._alerts.append(alert)
                            iter_alerts += 1
                            log.warning(
                                "ALERT: %s on %s (score=%.2f)",
                                alert["fault"],
                                alert["channel_id"],
                                alert["score"],
                            )
                            for sink in self.alert_sinks:
                                try:
                                    sink.publish(alert)
                                except Exception as exc:
                                    log.warning(
                                        "Alert sink %s failed: %s", type(sink).__name__, exc
                                    )

                    # Reset consecutive error counter on success
                    self._consecutive_errors = 0

                except Exception as exc:
                    self._consecutive_errors += 1
                    self.stats.total_errors += 1
                    log.error(
                        "Iteration %d error (%d/%d consecutive): %s",
                        iteration,
                        self._consecutive_errors,
                        self.cfg.max_consecutive_errors,
                        exc,
                    )
                    if self._consecutive_errors >= self.cfg.max_consecutive_errors:
                        log.critical(
                            "Max consecutive errors reached (%d) — aborting",
                            self.cfg.max_consecutive_errors,
                        )
                        break
                    # Skip the rest of this iteration's bookkeeping
                    iteration += 1
                    if max_iterations and iteration >= max_iterations:
                        break
                    continue

                # --- Per-iteration metrics ---
                iter_ms = (time.monotonic() - iter_start) * 1000
                iter_stat = IterationStats(
                    iteration=iteration,
                    rows_ingested=len(batch),
                    inference_ms=round(iter_ms, 2),
                    alerts_emitted=iter_alerts,
                    buffer_rows=len(self._buffer),
                    memory_rss_mb=round(self._get_rss_mb(), 1),
                )
                self.stats.history.append(iter_stat)
                self.stats.total_iterations += 1
                self.stats.total_rows += len(batch)
                self.stats.total_alerts += iter_alerts
                self.stats.total_inference_ms += iter_ms

                log.debug(
                    "iter=%d  rows=%d  infer=%.1fms  alerts=%d  buf=%d  rss=%.1fMB",
                    iteration,
                    len(batch),
                    iter_ms,
                    iter_alerts,
                    len(self._buffer),
                    iter_stat.memory_rss_mb,
                )

                # --- Heartbeat ---
                iteration += 1
                if self.cfg.heartbeat_interval > 0 and iteration % self.cfg.heartbeat_interval == 0:
                    self._write_heartbeat(iteration)

                # --- Model hot-reload ---
                self._check_model_reload()

                if max_iterations and iteration >= max_iterations:
                    break

        finally:
            # Graceful cleanup regardless of how we exit
            self._restore_signal_handlers()
            self._flush_scored()
            # Close any open cycle
            if self._cycle_accumulator is not None:
                s = self._cycle_accumulator.close()
                if s:
                    self.cycle_summaries.append(s)
                    if self._lifetime_tracker is not None:
                        self._lifetime_tracker.ingest(s)
                self.cycle_summaries.extend(
                    s2 for s2 in self._cycle_accumulator.completed if s2 not in self.cycle_summaries
                )
            if self.writer and self._alerts:
                self.writer.write_alerts(self._alerts)
            for sink in self.alert_sinks:
                try:
                    sink.close()
                except Exception as exc:
                    log.warning("Alert sink close failed: %s", exc)
            self._write_heartbeat(iteration)

        log.info(
            "Edge runtime stopped — %d alerts, %d iterations, %.0f ms avg inference, %d errors",
            len(self._alerts),
            self.stats.total_iterations,
            self.stats.avg_inference_ms,
            self.stats.total_errors,
        )
        return self._alerts

    def stop(self) -> None:
        self._running = False

    def _flush_scored(self) -> None:
        """Write accumulated scored data to disk and free memory."""
        if not self.writer or not self._scored_accumulator:
            return
        all_scored = pd.concat(self._scored_accumulator, ignore_index=True)
        name = f"scored_part{self._flush_part:04d}"
        self.writer.write_scored(all_scored, name=name)
        self._flush_part += 1
        self._scored_accumulator.clear()
        self._flush_counter = 0
        log.info("Flushed %d scored rows to disk (part %d)", len(all_scored), self._flush_part - 1)

    @staticmethod
    def _make_alert(row: pd.Series) -> dict:
        ts = row.get("timestamp", datetime.now(tz=timezone.utc))
        if isinstance(ts, pd.Timestamp):
            ts = ts.isoformat()
        else:
            ts = str(ts)

        return {
            "timestamp": ts,
            "channel_id": row.get("channel_id", "unknown"),
            "score": float(row.get("anomaly_score", 0)),
            "fault": row.get("predicted_fault", "none"),
            "confidence": float(row.get("fault_confidence", 0)),
            "causes": row.get("likely_causes", []),
        }
