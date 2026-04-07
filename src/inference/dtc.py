"""DTC debounce / healing state machine.

In production OEM software, a fault classification result does NOT immediately
become a Diagnostic Trouble Code (DTC).  The AUTOSAR Dem (Diagnostic Event
Manager) runs each diagnostic event through a debounce / healing counter before
changing the DTC status.

Two counter-based rules gate DTC confirmation and clearance:

    PENDING  → CONFIRMED  after ``fail_threshold`` consecutive failing evaluations
    CONFIRMED → ABSENT    after ``heal_threshold`` consecutive passing evaluations

This means:
  - A single transient spike that passes once is silently discarded — it never
    leaves PENDING state before the heal counter resets it.
  - A genuine sustained fault must persist for N consecutive evaluations before
    an alert is published.
  - Once confirmed, the fault must stay clean for M consecutive evaluations
    before it is declared healed.

Production calibration (AUTOSAR Dem counter-based debouncing):
  Typical values: fail_threshold = 3–5, heal_threshold = 10–20.

State transitions::

    ┌──────────┐  fail (n++)       ┌─────────┐  fail (n >= N)  ┌───────────┐
    │  ABSENT  │ ──────────────── ►│ PENDING │ ──────────────► │ CONFIRMED │
    │          │◄──────────────── │         │  pass (reset)   │           │
    └──────────┘  pass (reset)    └─────────┘                 └───────────┘
                                                                 │       ▲
                                                           pass  │       │  fail
                                                          (m++)  ▼       │  (reset m)
                                                               HEALING ──┘  if m >= M → ABSENT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.schemas.telemetry import FaultType


class DTCStatus(str, Enum):
    """DTC status aligned to AUTOSAR Dem DTC status bits (simplified)."""

    ABSENT = "absent"  # No fault observed or fault healed
    PENDING = "pending"  # Fault observed but not yet confirmed
    CONFIRMED = "confirmed"  # Fault confirmed — DTC asserted, alert may be emitted
    HEALING = "healing"  # Previously confirmed, now passing but not yet cleared


@dataclass
class _DTCRecord:
    """Internal state for one (channel_id, fault_type) pair."""

    status: DTCStatus = DTCStatus.ABSENT
    fail_count: int = 0  # consecutive failing evals
    heal_count: int = 0  # consecutive passing evals while CONFIRMED


class DTCDebouncer:
    """Counter-based DTC debounce and healing state machine.

    One ``DTCDebouncer`` instance is shared across all channels in the edge
    runtime.  It maintains per (channel_id, fault_type) state internally.

    Parameters
    ----------
    fail_threshold:
        Number of consecutive failing evaluations required to move a fault from
        PENDING to CONFIRMED.  Default: 3.
    heal_threshold:
        Number of consecutive passing evaluations required to move a confirmed
        fault from HEALING to ABSENT (cleared).  Default: 10.
    """

    def __init__(self, fail_threshold: int = 3, heal_threshold: int = 10) -> None:
        if fail_threshold < 1:
            raise ValueError(f"fail_threshold must be >= 1, got {fail_threshold}")
        if heal_threshold < 1:
            raise ValueError(f"heal_threshold must be >= 1, got {heal_threshold}")
        self.fail_threshold = fail_threshold
        self.heal_threshold = heal_threshold
        self._records: dict[tuple[str, str], _DTCRecord] = {}

    def update(self, channel_id: str, fault_type: FaultType, fault_present: bool) -> DTCStatus:
        """Advance the state machine for one (channel, fault) evaluation.

        Parameters
        ----------
        channel_id: Channel identifier.
        fault_type: The fault being evaluated.  NONE is always absent.
        fault_present: True = fault classifier detected this fault this cycle.

        Returns
        -------
        DTCStatus after applying this evaluation.
        """
        if fault_type == FaultType.NONE or not fault_present:
            return self._handle_pass(channel_id, fault_type)
        return self._handle_fail(channel_id, fault_type)

    def status(self, channel_id: str, fault_type: FaultType) -> DTCStatus:
        """Return the current DTC status without advancing state."""
        key = (channel_id, fault_type.value)
        return self._records.get(key, _DTCRecord()).status

    def is_confirmed(self, channel_id: str, fault_type: FaultType) -> bool:
        """Return True only for CONFIRMED faults — the publishable alert gate."""
        return self.status(channel_id, fault_type) == DTCStatus.CONFIRMED

    def reset_channel(self, channel_id: str) -> None:
        """Clear all DTC state for a channel (e.g. on ignition cycle)."""
        self._records = {k: v for k, v in self._records.items() if k[0] != channel_id}

    def reset_all(self) -> None:
        """Clear all DTC state (e.g. on runtime restart)."""
        self._records.clear()

    def snapshot(self) -> dict[str, dict]:
        """Return a serialisable snapshot of all non-ABSENT DTC records."""
        return {
            f"{ch}|{ft}": {
                "status": rec.status.value,
                "fail_count": rec.fail_count,
                "heal_count": rec.heal_count,
            }
            for (ch, ft), rec in self._records.items()
            if rec.status != DTCStatus.ABSENT
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(self, channel_id: str, fault_type: FaultType) -> _DTCRecord:
        key = (channel_id, fault_type.value)
        if key not in self._records:
            self._records[key] = _DTCRecord()
        return self._records[key]

    def _handle_fail(self, channel_id: str, fault_type: FaultType) -> DTCStatus:
        rec = self._record(channel_id, fault_type)
        if rec.status == DTCStatus.ABSENT:
            rec.status = DTCStatus.PENDING
            rec.fail_count = 1
            rec.heal_count = 0
        elif rec.status == DTCStatus.PENDING:
            rec.fail_count += 1
            if rec.fail_count >= self.fail_threshold:
                rec.status = DTCStatus.CONFIRMED
        elif rec.status == DTCStatus.CONFIRMED:
            rec.heal_count = 0  # any fail resets healing progress
        elif rec.status == DTCStatus.HEALING:
            # Fault returned — back to CONFIRMED, reset heal counter
            rec.status = DTCStatus.CONFIRMED
            rec.heal_count = 0
        return rec.status

    def _handle_pass(self, channel_id: str, fault_type: FaultType) -> DTCStatus:
        key = (channel_id, fault_type.value)
        if key not in self._records:
            return DTCStatus.ABSENT
        rec = self._records[key]
        if rec.status == DTCStatus.ABSENT:
            pass  # already clear
        elif rec.status == DTCStatus.PENDING:
            # Fault healed before confirmation — silently discard
            rec.status = DTCStatus.ABSENT
            rec.fail_count = 0
        elif rec.status == DTCStatus.CONFIRMED:
            rec.status = DTCStatus.HEALING
            rec.heal_count = 1
        elif rec.status == DTCStatus.HEALING:
            rec.heal_count += 1
            if rec.heal_count >= self.heal_threshold:
                rec.status = DTCStatus.ABSENT
                rec.fail_count = 0
                rec.heal_count = 0
        return rec.status
