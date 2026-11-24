"""Scorer logic without Kafka or HTTP, so it can be tested directly.

A batch of raw Kafka values goes through three steps:
  1. parse + validate each value; failures become DLQ records (never crash the loop)
  2. compute features per card with the shared FeatureEngine (state lives here)
  3. after the model answers, join scores back into ScoredEvents
"""

from __future__ import annotations

import base64
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import orjson
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from confluent_kafka import TopicPartition
from pydantic import ValidationError

from fraud_common.features import FeatureEngine, category_code
from fraud_common.schemas import (
    CardProfile,
    ScoreBatchRequest,
    ScoreBatchResponse,
    ScoredEvent,
    ScoreItem,
    TransactionEvent,
)


@dataclass(frozen=True, slots=True)
class DeadLetter:
    key: bytes | None
    value: bytes

    @classmethod
    def build(cls, key: bytes | None, raw: bytes | None, error: str, stage: str) -> DeadLetter:
        payload = {
            "raw": base64.b64encode(raw or b"").decode(),
            "error": error[:2000],
            "stage": stage,
            "failed_at": datetime.now(UTC).isoformat(),
        }
        return cls(key=key, value=orjson.dumps(payload))


@dataclass(slots=True)
class PreparedBatch:
    events: list[TransactionEvent] = field(default_factory=list)
    request: ScoreBatchRequest | None = None
    dead: list[DeadLetter] = field(default_factory=list)


def load_profiles(data_dir: Path) -> dict[str, CardProfile]:
    path = data_dir / "profiles.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make data` first")
    rows = pq.read_table(path).to_pylist()
    profiles = [CardProfile.model_validate(r) for r in rows]
    return {p.card_id: p for p in profiles}


HISTORY_CHUNK = 50_000
RESTORE_IDLE_POLLS = 5  # give up after 5 empty 1-second polls
HISTORY_COLUMNS = ("card_id", "event_time", "amount", "category", "merch_lat", "merch_long")


def load_history(data_dir: Path, engine: FeatureEngine) -> int:
    """Feed every training-period transaction through `engine`, oldest first.

    The model was trained and evaluated on features computed over each card's whole
    history, so the live scorer starts from the same history. Without it, lifetime stats
    such as a card's average amount or "first time in this category" are computed from
    a few days and false alarms roughly double (measured: precision 0.889 vs 0.940).
    Returns the number of transactions loaded."""
    path = data_dir / "transactions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make data` first")
    table = pq.read_table(
        path, columns=list(HISTORY_COLUMNS), filters=[("split_source", "=", "train")]
    )
    # Same order as training (event time; stable, so file order breaks ties).
    table = table.take(pc.sort_indices(table["event_time"]))
    process = engine.process_raw
    # Python objects for 1.3M rows at once would need ~0.5 GB, so convert in chunks.
    for chunk in table.to_batches(max_chunksize=HISTORY_CHUNK):
        micros = pc.cast(chunk["event_time"].cast(pa.timestamp("us", tz="UTC")), pa.int64())
        for card, us, amount, category, lat, lon in zip(
            chunk["card_id"].to_pylist(),
            micros.to_pylist(),
            chunk["amount"].to_pylist(),
            chunk["category"].to_pylist(),
            chunk["merch_lat"].to_pylist(),
            chunk["merch_long"].to_pylist(),
            strict=True,
        ):
            process(
                card_id=card, ts=us / 1e6, amount=amount, category=category_code(category),
                merch_lat=lat, merch_long=lon,
            )  # fmt: skip
    return int(table.num_rows)


class PartitionReader(Protocol):
    """The parts of confluent_kafka.Consumer that `restore_state` uses."""

    def get_watermark_offsets(
        self, partition: TopicPartition, timeout: float = ...
    ) -> tuple[int, int]: ...
    def assign(self, partitions: list[TopicPartition]) -> None: ...
    def consume(self, num_messages: int = ..., timeout: float = ...) -> list[Any]: ...


def restore_state(
    engine: FeatureEngine,
    reader: PartitionReader,
    committed: dict[tuple[str, int], int],
    max_per_partition: int,
) -> int:
    """Re-read each partition up to its committed offset and feed the events (in order,
    without scoring) into `engine`, so a restarted scorer remembers what each card did
    since the training-period history. Returns the number of events replayed."""
    targets: dict[tuple[str, int], int] = {}
    starts = []
    for (topic, partition), end in committed.items():
        if end <= 0:
            continue
        low, _high = reader.get_watermark_offsets(TopicPartition(topic, partition), timeout=10)
        start = max(low, end - max_per_partition)
        if start < end:
            targets[(topic, partition)] = end
            starts.append(TopicPartition(topic, partition, start))
    if not targets:
        return 0
    reader.assign(starts)
    replayed = idle = 0
    while targets and idle < RESTORE_IDLE_POLLS:
        messages = reader.consume(num_messages=1000, timeout=1.0)
        idle = 0 if messages else idle + 1
        for m in messages:
            key = (m.topic(), m.partition())
            stop_at = targets.get(key)
            if m.error() is not None or stop_at is None:
                continue
            if m.offset() >= stop_at - 1:
                del targets[key]  # this is the partition's last message to replay
                if m.offset() >= stop_at:
                    continue
            try:
                event = TransactionEvent.model_validate_json(m.value() or b"")
            except ValidationError:
                continue  # it went to the DLQ the first time, too
            k = m.key()
            if k is None or k.decode(errors="replace") == event.card_id:
                engine.process(event)
                replayed += 1
    return replayed


def _clean(value: float) -> float | None:
    return None if math.isnan(value) else value


class ScoringCore:
    def __init__(
        self,
        engine: FeatureEngine,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.engine = engine
        self._clock_ns = clock_ns

    def prepare(self, messages: Sequence[tuple[bytes | None, bytes | None]]) -> PreparedBatch:
        """(key, value) pairs in partition order -> events + model request + dead letters."""
        batch = PreparedBatch()
        items: list[ScoreItem] = []
        for key, value in messages:
            if value is None:
                batch.dead.append(DeadLetter.build(key, value, "empty message", "parse"))
                continue
            try:
                event = TransactionEvent.model_validate_json(value)
            except ValidationError as exc:
                reason = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
                )
                batch.dead.append(DeadLetter.build(key, value, reason, "validate"))
                continue
            if key is not None and key.decode(errors="replace") != event.card_id:
                batch.dead.append(
                    DeadLetter.build(key, value, "message key does not match card_id", "validate")
                )
                continue
            features = self.engine.process(event)
            batch.events.append(event)
            items.append(ScoreItem(tx_id=event.tx_id, features=[_clean(v) for v in features]))
        if items:
            batch.request = ScoreBatchRequest(items=items)
        return batch

    def finish(
        self, events: Sequence[TransactionEvent], response: ScoreBatchResponse
    ) -> list[ScoredEvent]:
        if len(response.results) != len(events):
            raise ValueError("model returned a different number of results")
        scored_at = self._clock_ns()
        out = []
        for event, result in zip(events, response.results, strict=True):
            if result.tx_id != event.tx_id:
                raise ValueError("model results are out of order")
            out.append(
                ScoredEvent(
                    tx_id=event.tx_id,
                    card_id=event.card_id,
                    merchant=event.merchant,
                    category=event.category,
                    amount=event.amount,
                    event_time=event.event_time,
                    score=result.score,
                    is_flagged=result.is_flagged,
                    threshold=response.threshold,
                    model_version=response.model_version,
                    produced_at_ns=event.produced_at_ns,
                    scored_at_ns=scored_at,
                )
            )
        return out
