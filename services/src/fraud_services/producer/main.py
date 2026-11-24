"""Transaction producer: replays held-out card transactions into Kafka at a fixed rate.

Only the held-out test period is replayed. The scorer loads every card's earlier history
(the training period) at startup, so live features match the ones the model was evaluated
on. Each transaction's true label is published to
`transactions.labels` `label_delay_s` later, like a chargeback. With `loop` on, the data
repeats forever with event times shifted forward and fresh transaction ids.
"""

from __future__ import annotations

import hashlib
import logging
import signal
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from confluent_kafka import Consumer, KafkaError, Message, Producer, TopicPartition
from prometheus_client import Counter, Gauge, start_http_server
from pydantic import ValidationError

from fraud_common.kafka import TOPIC_LABELS, TOPIC_RAW, client_config
from fraud_common.logging_config import configure_logging
from fraud_common.schemas import LabelEvent, TransactionEvent
from fraud_common.settings import KafkaSettings, LoggingSettings, ProducerSettings

log = logging.getLogger(__name__)

PRODUCED = Counter("producer_transactions_total", "Transactions produced")
LABELS = Counter("producer_labels_total", "Labels produced")
FAILED = Counter("producer_delivery_failures_total", "Kafka delivery failures")
TARGET = Gauge("producer_target_tps", "Configured rate")
METRICS_PORT = 9102


class Pacer:
    """Holds a steady average rate: event i is due at start + i / rate."""

    def __init__(
        self,
        rate: float,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self._interval = 1.0 / rate
        self._clock, self._sleep = clock, sleep
        self._start = clock()
        self._count = 0

    def wait(self) -> None:
        due = self._start + self._count * self._interval
        ahead = due - self._clock()
        if ahead > 0.001:  # sleep only when worth it; tiny gaps are absorbed in bursts
            self._sleep(ahead)
        elif ahead < -1.0:
            # More than 1 s behind (e.g. broker stall): re-anchor instead of bursting.
            self._start = self._clock() - self._count * self._interval
        self._count += 1


@dataclass(frozen=True, slots=True)
class Replay:
    rows: pa.Table
    span: timedelta
    first: datetime

    @classmethod
    def load(cls, data_dir: Path) -> Replay:
        path = data_dir / "transactions.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run `make data` first")
        table = pq.read_table(path)
        rows = table.filter(pc.equal(table["split_source"], "test"))
        if rows.num_rows == 0:
            raise ValueError("no test rows in transactions.parquet")
        first = pc.min(rows["event_time"]).as_py()
        last = pc.max(rows["event_time"]).as_py()
        return cls(rows=rows, span=last - first + timedelta(days=1), first=first)

    def lap_of(self, last_seen: datetime | None) -> int:
        """Which lap `last_seen` (the newest event already in Kafka) belongs to."""
        if last_seen is None or last_seen < self.first:
            return 0
        return (last_seen - self.first) // self.span

    def iterate(
        self, loop: bool, after: datetime | None = None
    ) -> Iterator[tuple[TransactionEvent, bool]]:
        """(event without produced_at, is_fraud), one pass or forever if `loop`.

        With `after` (the newest event already in Kafka), the stream resumes right after
        it: a second `make demo` continues where the first stopped, so the scorer sees one
        continuous history and live scores keep matching the offline evaluation. Each lap
        after the first shifts event times forward by `span` and gets fresh tx_ids."""
        lap = self.lap_of(after)
        while True:
            shift = self.span * lap
            rows = self.rows
            if after is not None:
                cutoff = pa.scalar(after - shift, type=rows["event_time"].type)
                rows = rows.filter(pc.greater(rows["event_time"], cutoff))
                after = None
                if rows.num_rows == 0:  # the previous run finished this lap
                    lap += 1
                    continue
            for batch in rows.to_batches(max_chunksize=10_000):
                for r in batch.to_pylist():
                    tx_id = r["tx_id"] if lap == 0 else _lap_id(r["tx_id"], lap)
                    event = TransactionEvent(
                        tx_id=tx_id,
                        card_id=r["card_id"],
                        merchant=r["merchant"],
                        category=r["category"],
                        amount=r["amount"],
                        event_time=r["event_time"] + shift,
                        merch_lat=r["merch_lat"],
                        merch_long=r["merch_long"],
                        produced_at_ns=0,
                    )
                    yield event, bool(r["is_fraud"])
            if not loop:
                return
            lap += 1


def _lap_id(tx_id: str, lap: int) -> str:
    return hashlib.sha256(f"{tx_id}:{lap}".encode()).hexdigest()[:32]


def read_last_event_time(consumer: Consumer, timeout_s: float = 10.0) -> datetime | None:
    """Newest event_time among the last message of each transactions.raw partition."""
    meta = consumer.list_topics(TOPIC_RAW.name, timeout=timeout_s)
    topic = meta.topics.get(TOPIC_RAW.name)
    if topic is None or topic.error is not None:
        raise RuntimeError(f"cannot read {TOPIC_RAW.name} metadata")
    tails = []
    for p in topic.partitions:
        low, high = consumer.get_watermark_offsets(
            TopicPartition(TOPIC_RAW.name, p), timeout=timeout_s
        )
        if high > low:
            tails.append(TopicPartition(TOPIC_RAW.name, p, high - 1))
    if not tails:
        return None
    consumer.assign(tails)
    newest: datetime | None = None
    remaining = len(tails)
    deadline = time.monotonic() + timeout_s
    while remaining and time.monotonic() < deadline:
        msg = consumer.poll(0.5)
        if msg is None or msg.error() is not None:
            continue
        remaining -= 1
        try:
            seen = TransactionEvent.model_validate_json(msg.value() or b"").event_time
        except ValidationError:
            continue  # a malformed tail message (it went to the DLQ); ignore it
        newest = seen if newest is None or seen > newest else newest
    consumer.unassign()
    return newest


class LabelQueue:
    """Labels wait `delay_s`, then go out in order (every label has the same delay)."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self._pending: deque[tuple[float, LabelEvent]] = deque()

    def __len__(self) -> int:
        return len(self._pending)

    def add(self, now: float, event: TransactionEvent, is_fraud: bool) -> None:
        label = LabelEvent(
            tx_id=event.tx_id,
            card_id=event.card_id,
            is_fraud=is_fraud,
            labelled_at=datetime.fromtimestamp(now + self.delay_s, tz=UTC),
        )
        self._pending.append((now + self.delay_s, label))

    def due(self, now: float, *, flush_all: bool = False) -> list[LabelEvent]:
        out = []
        while self._pending and (flush_all or self._pending[0][0] <= now):
            out.append(self._pending.popleft()[1])
        return out


class TransactionProducer:
    def __init__(
        self,
        settings: ProducerSettings,
        kafka: KafkaSettings,
        *,
        replay: Replay | None = None,
        producer: Producer | None = None,
        last_seen: datetime | None = None,
    ) -> None:
        """Collaborators default to the real ones; tests inject in-memory fakes."""
        self.settings = settings
        self.replay = replay or Replay.load(Path(settings.data_dir))
        self.labels = LabelQueue(settings.label_delay_s)
        self.kafka = producer or Producer(
            client_config(kafka, "producer")
            | {"enable.idempotence": True, "acks": "all", "linger.ms": 1,
               "compression.type": "lz4", "queue.buffering.max.messages": 500_000}
        )  # fmt: skip
        self._running = True
        self.last_seen = last_seen

    def stop(self, *_: object) -> None:
        self._running = False

    @staticmethod
    def _on_delivery(err: KafkaError | None, _msg: Message) -> None:
        if err is not None:
            FAILED.inc()
            log.error("delivery failed", extra={"error": str(err)})

    def _send(self, topic: str, key: str, value: bytes) -> None:
        while True:
            try:
                self.kafka.produce(
                    topic, key=key.encode(), value=value, on_delivery=self._on_delivery
                )
            except BufferError:  # local queue full: let librdkafka drain, then retry
                self.kafka.poll(0.05)
            else:
                return

    def _send_labels(self, now: float, *, flush_all: bool = False) -> None:
        for label in self.labels.due(now, flush_all=flush_all):
            self._send(TOPIC_LABELS.name, label.card_id, label.model_dump_json().encode())
            LABELS.inc()

    def run(self) -> int:
        TARGET.set(self.settings.rate_tps)
        pacer = Pacer(self.settings.rate_tps)
        sent = 0
        log.info("producer started", extra={"rows": self.replay.rows.num_rows,
                                             "rate_tps": self.settings.rate_tps})  # fmt: skip
        log.info("replay position", extra={"lap": self.replay.lap_of(self.last_seen),
                                            "after": str(self.last_seen)})  # fmt: skip
        for event, is_fraud in self.replay.iterate(self.settings.loop, after=self.last_seen):
            if not self._running or (self.settings.max_events and sent >= self.settings.max_events):
                break
            pacer.wait()
            stamped = event.model_copy(update={"produced_at_ns": time.time_ns()})
            self._send(TOPIC_RAW.name, stamped.card_id, stamped.model_dump_json().encode())
            now = time.time()
            self.labels.add(now, stamped, is_fraud)
            self._send_labels(now)
            sent += 1
            PRODUCED.inc()
            if sent % 1000 == 0:
                self.kafka.poll(0)
        # Let outstanding labels mature, then deliver everything.
        deadline = time.time() + self.labels.delay_s
        while self._running and len(self.labels) and time.time() < deadline:
            self._send_labels(time.time())
            time.sleep(0.05)
        self._send_labels(time.time(), flush_all=True)
        left = self.kafka.flush(30)
        log.info("producer finished", extra={"sent": sent, "undelivered": left})
        return 0 if left == 0 else 1


def main() -> None:
    configure_logging(LoggingSettings(), service="producer")
    start_http_server(METRICS_PORT)
    kafka = KafkaSettings()
    reader = Consumer(client_config(kafka, "producer") | {
        "group.id": "producer-tail-reader", "enable.auto.commit": False})  # fmt: skip
    try:
        last_seen = read_last_event_time(reader)
    finally:
        reader.close()
    producer = TransactionProducer(ProducerSettings(), kafka, last_seen=last_seen)

    def handle(_sig: int, _frame: FrameType | None) -> None:
        producer.stop()

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    raise SystemExit(producer.run())


if __name__ == "__main__":
    main()
