"""Stream scorer: transactions.raw -> features -> model API -> transactions.scored.

Pipelined micro-batches. While the model API scores batch N, the scorer is already
reading and preparing batch N+1, and Kafka delivers earlier results in the background.
Up to `max_inflight` batches wait on the model at once; results are always written in
the order the batches were read.

Delivery is at-least-once: a batch's offsets are stored only after every scored event
and dead letter in it (and in every earlier batch) is acknowledged by the broker.
Topics are keyed by card_id, so each card's events reach one scorer, in order, and its
feature state lives in this process.
"""

from __future__ import annotations

import gc
import logging
import signal
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

from confluent_kafka import Consumer, KafkaError, Message, Producer, TopicPartition
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from fraud_common.features import FeatureEngine
from fraud_common.kafka import TOPIC_DLQ, TOPIC_RAW, TOPIC_SCORED, client_config, next_offsets
from fraud_common.logging_config import configure_logging
from fraud_common.schemas import ScoreBatchResponse
from fraud_common.settings import KafkaSettings, LoggingSettings, ScorerSettings
from fraud_services.scorer.client import (
    Backoff,
    ModelClient,
    ModelUnavailableError,
    score_with_retry,
)
from fraud_services.scorer.core import (
    PreparedBatch,
    ScoringCore,
    load_history,
    load_profiles,
    restore_state,
)

log = logging.getLogger(__name__)

CONSUMED = Counter("scorer_consumed_total", "Messages read from transactions.raw")
SCORED = Counter("scorer_scored_total", "Events written to transactions.scored")
FLAGGED = Counter("scorer_flagged_total", "Scored events flagged as fraud")
DEAD = Counter("scorer_dead_letters_total", "Messages sent to the DLQ")
RETRIES = Counter("scorer_model_retries_total", "Model API retries")
CARDS = Gauge("scorer_cards_in_state", "Cards with feature state in memory")
INFLIGHT = Gauge("scorer_batches_inflight", "Batches waiting on the model API")
UNACKED = Gauge("scorer_batches_unacked", "Scored batches waiting for broker acks")
E2E = Histogram(
    "scorer_e2e_latency_seconds",
    "Produced to scored, per event",
    # Fine-grained below 50 ms so p95 can be read to about a millisecond.
    buckets=(0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.012,
             0.015, 0.018, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.1, 0.25, 0.5, 1, 2.5, 5),
)  # fmt: skip
FETCH = Histogram(
    "scorer_fetch_latency_seconds",
    "Produced to read by the scorer, per event (the Kafka part of the end-to-end time)",
    buckets=(0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.015, 0.020, 0.030,
             0.050, 0.1, 0.25, 0.5, 1, 2.5, 5),
)  # fmt: skip
BATCH = Histogram(
    "scorer_batch_size", "Messages per micro-batch", buckets=(1, 8, 32, 64, 128, 256, 512)
)
MODEL_CALL = Histogram(
    "scorer_model_call_seconds",
    "Model API round trip per batch",
    buckets=(0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.1, 1),
)

MAX_UNACKED = 64  # scored batches awaiting broker acks before the loop waits for them


@dataclass(slots=True)
class InFlight:
    """A batch whose model call is running."""

    records: list[Message]
    batch: PreparedBatch
    future: Future[ScoreBatchResponse] | None
    started: float


@dataclass(slots=True)
class Unacked:
    """A scored batch whose output Kafka has not yet confirmed."""

    offsets: list[TopicPartition]
    pending: int = 0
    errors: list[str] = field(default_factory=list)

    def on_delivery(self, err: KafkaError | None, _msg: Message | None) -> None:
        self.pending -= 1
        if err is not None:
            self.errors.append(str(err))


class Scorer:
    def __init__(  # noqa: PLR0913 (injectable collaborators)
        self,
        settings: ScorerSettings,
        kafka: KafkaSettings,
        *,
        core: ScoringCore | None = None,
        client: ModelClient | None = None,
        consumer: Consumer | None = None,
        producer: Producer | None = None,
        restore_reader: Callable[[], Consumer] | None = None,
    ) -> None:
        """Collaborators default to the real ones; tests inject in-memory fakes."""
        self.settings = settings
        self.core = core or self._build_core(settings)
        self.client = client or ModelClient(
            str(settings.model_api_url).rstrip("/"),
            settings.api_key.get_secret_value(),
            timeout_s=settings.request_timeout_ms / 1000,
        )
        self.backoff = Backoff(max_s=settings.retry_max_backoff_ms / 1000)
        self.consumer = consumer or Consumer(
            client_config(kafka, "scorer")
            | {
                "group.id": settings.group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
                "enable.auto.offset.store": False,  # we store offsets after delivery
                "partition.assignment.strategy": "cooperative-sticky",
                "fetch.wait.max.ms": max(1, settings.batch_wait_ms),
            }
        )
        self.producer = producer or Producer(
            client_config(kafka, "scorer")
            | {
                "enable.idempotence": True,
                "acks": "all",
                "linger.ms": 1,
                "compression.type": "lz4",
            }
        )
        self._pool = ThreadPoolExecutor(settings.max_inflight, thread_name_prefix="model-call")
        self._inflight: deque[InFlight] = deque()
        self._unacked: deque[Unacked] = deque()
        self._running = True
        self._cards_by_partition: dict[int, set[str]] = {}
        self._restored = not settings.restore_state
        self._restore_reader = restore_reader or (
            lambda: Consumer(client_config(kafka, "scorer") | {
                "group.id": f"{settings.group_id}-restore", "enable.auto.commit": False})
        )  # fmt: skip

    @staticmethod
    def _build_core(settings: ScorerSettings) -> ScoringCore:
        data_dir = Path(settings.data_dir)
        engine = FeatureEngine(load_profiles(data_dir))
        if settings.bootstrap_history:
            started = time.perf_counter()
            rows = load_history(data_dir, engine)
            log.info("card history loaded", extra={
                "rows": rows, "cards": engine.card_count,
                "seconds": round(time.perf_counter() - started, 1)})  # fmt: skip
        # The loaded history is long-lived. Moving it out of the garbage collector's view
        # stops every full collection from re-scanning it (halves the scorer's p99 work).
        gc.collect()
        gc.freeze()
        return ScoringCore(engine)

    def stop(self, *_: object) -> None:
        self._running = False

    # ------------------------------------------------------------------ rebalancing

    def _on_assign(self, consumer: Consumer, partitions: list[TopicPartition]) -> None:
        """On the first assignment after start, rebuild card state from the events this
        group already scored (they are before the committed offsets)."""
        if self._restored or not partitions:
            return
        self._restored = True
        started = time.perf_counter()
        committed = {
            (tp.topic, tp.partition): tp.offset
            for tp in consumer.committed(partitions, timeout=30)
            if tp.offset is not None and tp.offset > 0
        }
        reader = self._restore_reader()
        try:
            replayed = restore_state(self.core.engine, reader, committed,
                                     self.settings.restore_max_per_partition)  # fmt: skip
        finally:
            reader.close()
        log.info("card state restored", extra={
            "events": replayed, "partitions": len(committed),
            "seconds": round(time.perf_counter() - started, 1)})  # fmt: skip

    def _on_revoke(self, _consumer: Consumer, partitions: list[TopicPartition]) -> None:
        """Finish and commit everything read so far, then drop the revoked cards' state:
        another scorer owns them now, and stale history would skew their features."""
        self.drain()
        dropped = 0
        for tp in partitions:
            cards = self._cards_by_partition.pop(tp.partition, set())
            dropped += self.core.engine.forget(cards)
        log.info("partitions revoked", extra={"partitions": [tp.partition for tp in partitions],
                                               "cards_dropped": dropped})  # fmt: skip

    def _keep_alive(self) -> None:
        # Paused partitions return nothing, but polling keeps the group membership alive.
        self.consumer.poll(0)
        self.producer.poll(0)

    # ------------------------------------------------------------------ main loop

    def run(self) -> None:
        self.consumer.subscribe(
            [TOPIC_RAW.name],
            on_assign=self._on_assign,
            on_revoke=self._on_revoke,
            on_lost=self._on_revoke,
        )
        log.info("scorer started", extra={"batch_max": self.settings.batch_max,
                                           "max_inflight": self.settings.max_inflight})  # fmt: skip
        try:
            while self._running:
                self.step()
            self.drain()
        finally:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self.producer.flush(10)
            self.consumer.close()
            self.client.close()

    def step(self) -> None:
        """One turn: read a batch (without waiting if the model is busy), hand it to the
        model, then write out whatever the model has finished, oldest first."""
        wait_s = 0.0 if self._inflight else self.settings.batch_wait_ms / 1000
        messages = self.consumer.consume(num_messages=self.settings.batch_max, timeout=wait_s)
        if messages:
            self.submit(messages)
        # Block on the oldest batch when the pipeline is full, or when there is nothing
        # new to read (rather than spinning on empty polls).
        must_wait = len(self._inflight) >= self.settings.max_inflight or not messages
        while self._inflight and (must_wait or self._done(self._inflight[0])):
            self.complete(self._inflight.popleft())
            must_wait = False
        self.producer.poll(0)
        self.store_acked()
        INFLIGHT.set(len(self._inflight))
        UNACKED.set(len(self._unacked))

    @staticmethod
    def _done(item: InFlight) -> bool:
        return item.future is None or item.future.done()

    def submit(self, messages: Sequence[Message]) -> None:
        records = [m for m in messages if m.error() is None]
        for m in messages:
            if (err := m.error()) is not None and err.code() != KafkaError._PARTITION_EOF:
                log.error("consumer error", extra={"error": str(err)})
        if not records:
            return
        CONSUMED.inc(len(records))
        BATCH.observe(len(records))
        batch = self.core.prepare([(m.key(), m.value()) for m in records])
        now_ns = time.time_ns()
        for event in batch.events:
            FETCH.observe(max(0, now_ns - event.produced_at_ns) / 1e9)
        for m in records:
            if (key := m.key()) is not None and (part := m.partition()) is not None:
                self._cards_by_partition.setdefault(part, set()).add(key.decode(errors="replace"))
        CARDS.set(self.core.engine.card_count)
        future = None
        if batch.request is not None:
            future = self._pool.submit(self.client.score, batch.request)
        self._inflight.append(InFlight(records, batch, future, time.perf_counter()))

    def complete(self, item: InFlight) -> None:
        """Write one batch's scores and dead letters; offsets follow once Kafka acks them."""
        batch = item.batch
        unacked = Unacked(offsets=next_offsets(item.records))
        if item.future is not None and batch.request is not None:
            try:
                response = item.future.result()
            except ModelUnavailableError:  # pause consumption, then retry with backoff
                assignment = self.consumer.assignment()
                self.consumer.pause(assignment)
                response = score_with_retry(
                    self.client, batch.request, self.backoff, self._keep_alive,
                    on_retry=lambda _e: RETRIES.inc(),
                )  # fmt: skip
                self.consumer.resume(assignment)
            MODEL_CALL.observe(time.perf_counter() - item.started)
            scored = self.core.finish(batch.events, response)
            for event in scored:
                self._produce(TOPIC_SCORED.name, event.card_id.encode(),
                              event.model_dump_json().encode(), unacked)  # fmt: skip
                E2E.observe(event.e2e_latency_ms / 1000)
            SCORED.inc(len(scored))
            FLAGGED.inc(sum(e.is_flagged for e in scored))
        for dead in batch.dead:
            self._produce(TOPIC_DLQ.name, dead.key, dead.value, unacked)
        DEAD.inc(len(batch.dead))
        self._unacked.append(unacked)
        while len(self._unacked) > MAX_UNACKED:  # bound memory if the broker is slow
            self.producer.poll(0.01)
            self.store_acked()

    def _produce(self, topic: str, key: bytes | None, value: bytes, unacked: Unacked) -> None:
        unacked.pending += 1
        while True:
            try:
                self.producer.produce(topic, key=key, value=value, on_delivery=unacked.on_delivery)
            except BufferError:  # local queue full: let librdkafka drain, then retry
                self.producer.poll(0.01)
            else:
                return

    def store_acked(self) -> None:
        """Store offsets for the leading batches whose output is fully acknowledged."""
        while self._unacked and self._unacked[0].pending == 0:
            done = self._unacked.popleft()
            if done.errors:
                # Not storing offsets means these messages are re-read after a restart.
                raise RuntimeError(f"delivery failed: {done.errors[:3]}")
            self.consumer.store_offsets(offsets=done.offsets)

    def drain(self) -> None:
        """Finish every in-flight batch and wait for all acks (shutdown, rebalance)."""
        while self._inflight:
            self.complete(self._inflight.popleft())
        remaining = self.producer.flush(30)
        self.store_acked()
        if remaining or self._unacked:
            raise RuntimeError(f"delivery timed out ({remaining} messages pending)")


def main() -> None:
    configure_logging(LoggingSettings(), service="scorer")
    settings = ScorerSettings()
    scorer = Scorer(settings, KafkaSettings())
    # Metrics (and so the health check) come up only after the card history is loaded,
    # so `docker compose up --wait` returns once the scorer can score at full speed.
    start_http_server(settings.metrics_port)

    def handle(_sig: int, _frame: FrameType | None) -> None:
        scorer.stop()

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    scorer.run()


if __name__ == "__main__":
    main()
