"""Live statistics for the dashboard, fed by the Kafka consumer thread.

Throughput and latency use a rolling window; precision and recall are cumulative since
start, computed by joining each scored event with its delayed ground-truth label (in
either arrival order). Delivery is at-least-once, so repeated tx_ids are ignored.
All public methods are thread-safe.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from fraud_common.schemas import ScoredEvent

MAX_TRACKED = 300_000  # recent tx_ids kept for joining and de-duplication (bounded memory)
MAX_RECENT_ALERTS = 200


@dataclass(frozen=True, slots=True)
class Snapshot:
    window_s: int
    tps: float
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    scored_total: int
    flagged_total: int
    labelled_total: int
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    model_version: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AlertNote:
    seq: int
    event: ScoredEvent


class LiveStats:
    def __init__(self, window_s: int, clock: Callable[[], float] = time.monotonic) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self.window_s = window_s
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: deque[tuple[float, float]] = deque()  # (arrival time, e2e latency ms)
        self._scored: OrderedDict[str, bool] = OrderedDict()  # tx_id -> flagged (recent)
        self._labelled: OrderedDict[str, bool] = OrderedDict()  # tx_id -> fraud (recent)
        self._unmatched: OrderedDict[str, bool] = OrderedDict()  # labels awaiting their score
        self.duplicates = 0
        self._alerts: deque[AlertNote] = deque(maxlen=MAX_RECENT_ALERTS)
        self._seq = 0
        self.scored_total = self.flagged_total = self.labelled_total = 0
        self.tp = self.fp = self.fn = 0
        self.model_version: str | None = None

    # ------------------------------------------------------------ writers

    def record_scored(self, event: ScoredEvent) -> bool:
        """Count a scored event. Returns False (and counts nothing) for a repeat."""
        now = self._clock()
        with self._lock:
            if event.tx_id in self._scored:
                self.duplicates += 1
                return False
            _bounded_put(self._scored, event.tx_id, event.is_flagged)
            self.scored_total += 1
            self.model_version = event.model_version
            self._recent.append((now, event.e2e_latency_ms))
            if event.is_flagged:
                self.flagged_total += 1
                self._seq += 1
                self._alerts.append(AlertNote(self._seq, event))
            label = self._unmatched.pop(event.tx_id, None)
            if label is not None:
                self._count(flagged=event.is_flagged, fraud=label)
            self._trim(now)
            return True

    def record_label(self, tx_id: str, is_fraud: bool) -> bool | None:
        """Returns whether the matching transaction was flagged, or None if not yet scored
        (or if this label is a repeat)."""
        with self._lock:
            if tx_id in self._labelled:
                self.duplicates += 1
                return None
            _bounded_put(self._labelled, tx_id, is_fraud)
            flagged = self._scored.get(tx_id)
            if flagged is None:
                _bounded_put(self._unmatched, tx_id, is_fraud)
                return None
            self._count(flagged=flagged, fraud=is_fraud)
            return flagged

    def flag_status(self, tx_id: str) -> bool | None:
        """Whether a recently scored transaction was flagged; None if not seen."""
        with self._lock:
            return self._scored.get(tx_id)

    def pending_label(self, tx_id: str) -> bool | None:
        """A label that arrived before its score, if any."""
        with self._lock:
            return self._unmatched.get(tx_id)

    # ------------------------------------------------------------ readers

    def snapshot(self) -> Snapshot:
        now = self._clock()
        with self._lock:
            self._trim(now)
            latencies = np.fromiter((lat for _, lat in self._recent), dtype=np.float64)
            p50 = p95 = p99 = None
            if latencies.size:
                p50, p95, p99 = (float(v) for v in np.percentile(latencies, [50, 95, 99]))
            flagged_labelled = self.tp + self.fp
            frauds = self.tp + self.fn
            return Snapshot(
                window_s=self.window_s,
                tps=latencies.size / self.window_s,
                p50_ms=p50,
                p95_ms=p95,
                p99_ms=p99,
                scored_total=self.scored_total,
                flagged_total=self.flagged_total,
                labelled_total=self.labelled_total,
                tp=self.tp,
                fp=self.fp,
                fn=self.fn,
                precision=self.tp / flagged_labelled if flagged_labelled else None,
                recall=self.tp / frauds if frauds else None,
                model_version=self.model_version,
            )

    def alerts_after(self, seq: int) -> list[AlertNote]:
        with self._lock:
            return [a for a in self._alerts if a.seq > seq]

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    # ------------------------------------------------------------ internals

    def _count(self, *, flagged: bool, fraud: bool) -> None:
        self.labelled_total += 1
        if flagged and fraud:
            self.tp += 1
        elif flagged:
            self.fp += 1
        elif fraud:
            self.fn += 1

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()


def _bounded_put(store: OrderedDict[str, bool], key: str, value: bool) -> None:
    store[key] = value
    if len(store) > MAX_TRACKED:
        store.popitem(last=False)
