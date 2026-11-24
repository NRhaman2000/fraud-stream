"""The Kafka loops of the scorer, producer and dashboard consumer, run against in-memory
fakes of the confluent-kafka Consumer/Producer. The real broker path is covered by
tests/integration/test_e2e.py."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import httpx
import orjson
import psycopg
import pytest
from confluent_kafka import KafkaError, TopicPartition

from fraud_common.features import FeatureEngine
from fraud_common.kafka import TOPIC_DLQ, TOPIC_LABELS, TOPIC_RAW, TOPIC_SCORED
from fraud_common.schemas import LabelEvent, ScoredEvent, TransactionEvent
from fraud_common.settings import (
    DashboardSettings,
    KafkaSettings,
    PostgresSettings,
    ProducerSettings,
    ScorerSettings,
)
from fraud_services.dashboard import consumer as dash_consumer
from fraud_services.dashboard.consumer import DashboardConsumer
from fraud_services.dashboard.stats import LiveStats
from fraud_services.producer.main import (
    Replay,
    TransactionProducer,
    _lap_id,
    read_last_event_time,
)
from fraud_services.scorer.client import ModelClient
from fraud_services.scorer.core import ScoringCore, restore_state
from fraud_services.scorer.main import Scorer
from fraud_training.ingest import ingest
from tests.fixtures.sparkov_like import write_kaggle_zip

CARD = "0123456789abcdef"
KEY = "k" * 40
PG = PostgresSettings(password="pw", app_password="app-password-123")


class FakeMsg:
    def __init__(
        self,
        topic: str,
        value: bytes | None,
        *,
        key: bytes | None = CARD.encode(),
        partition: int = 0,
        offset: int = 0,
        error: KafkaError | None = None,
    ) -> None:
        self._topic, self._value, self._key = topic, value, key
        self._partition, self._offset, self._error = partition, offset, error

    def topic(self) -> str:
        return self._topic

    def value(self) -> bytes | None:
        return self._value

    def key(self) -> bytes | None:
        return self._key

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset

    def error(self) -> KafkaError | None:
        return self._error


class FakeConsumer:
    """Hands out queued batches; calls `on_empty` once the queue runs dry."""

    def __init__(self, batches: list[list[FakeMsg]], on_empty: Callable[[], None]) -> None:
        self.batches = batches
        self.on_empty = on_empty
        self.stored: list[list[tuple[str, int, int]]] = []
        self.paused: list[Any] = []
        self.resumed: list[Any] = []
        self.subscribed: list[str] = []
        self.callbacks: dict[str, Any] = {}
        self.closed = False
        self.polls = 0
        self.seeks: list[tuple[str, int, int]] = []

    def subscribe(self, topics: list[str], **callbacks: Any) -> None:
        self.subscribed = topics
        self.callbacks = callbacks

    def consume(self, num_messages: int, timeout: float) -> list[FakeMsg]:
        assert num_messages >= 1
        assert timeout >= 0
        if self.batches:
            return self.batches.pop(0)
        self.on_empty()
        return []

    def assignment(self) -> list[TopicPartition]:
        return [TopicPartition(TOPIC_RAW.name, 0)]

    def pause(self, parts: Any) -> None:
        self.paused.append(parts)

    def resume(self, parts: Any) -> None:
        self.resumed.append(parts)

    def poll(self, _timeout: float) -> None:
        self.polls += 1

    def seek(self, tp: TopicPartition) -> None:
        self.seeks.append((tp.topic, tp.partition, tp.offset))

    def store_offsets(self, offsets: list[TopicPartition]) -> None:
        self.stored.append([(t.topic, t.partition, t.offset) for t in offsets])

    def close(self) -> None:
        self.closed = True


class FakeProducer:
    def __init__(self, *, fail_delivery: bool = False, buffer_errors: int = 0) -> None:
        self.sent: list[tuple[str, bytes | None, bytes]] = []
        self._pending: list[Callable[..., None]] = []
        self.fail_delivery = fail_delivery
        self.buffer_errors = buffer_errors
        self.polls = 0

    def produce(
        self, topic: str, *, key: bytes | None, value: bytes, on_delivery: Callable[..., None]
    ) -> None:
        if self.buffer_errors:
            self.buffer_errors -= 1
            raise BufferError("queue full")
        self.sent.append((topic, key, value))
        self._pending.append(on_delivery)

    def _deliver(self) -> None:
        err = KafkaError(KafkaError._MSG_TIMED_OUT) if self.fail_delivery else None
        for cb in self._pending:
            cb(err, None)
        self._pending.clear()

    def poll(self, _timeout: float) -> int:
        self.polls += 1
        self._deliver()
        return 0

    def flush(self, _timeout: float) -> int:
        self._deliver()
        return 0

    def topics(self) -> list[str]:
        return [t for t, _, _ in self.sent]


def raw_event(n: int, card: str = CARD) -> bytes:
    return orjson.dumps(
        {
            "schema_version": 2,
            "tx_id": f"{n:032x}",
            "card_id": card,
            "merchant": "Shop",
            "category": "grocery_pos",
            "amount": 12.5,
            "event_time": f"2020-06-21T12:{n % 60:02d}:00Z",
            "merch_lat": 40.0,
            "merch_long": -75.0,
            "produced_at_ns": 1_000_000,
        }
    )


def model_transport(fail_first: int = 0) -> httpx.MockTransport:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= fail_first:
            return httpx.Response(503, text="warming up")
        assert request.headers["X-API-Key"] == KEY
        items = orjson.loads(request.content)["items"]
        return httpx.Response(
            200,
            json={
                "model_version": "fraud_xgb-abcdefabcdef",
                "threshold": 0.5,
                "results": [
                    {"tx_id": it["tx_id"], "score": 0.9, "is_flagged": True} for it in items
                ],
            },
        )

    return httpx.MockTransport(handler)


def make_scorer(
    batches: list[list[FakeMsg]],
    *,
    fail_first: int = 0,
    fail_delivery: bool = False,
    buffer_errors: int = 0,
    max_inflight: int = 2,
) -> tuple[Scorer, FakeConsumer, FakeProducer]:
    env: dict[str, Any] = {
        "SCORER_API_KEY": KEY, "retry_max_backoff_ms": 100, "max_inflight": max_inflight,
    }  # fmt: skip
    settings = ScorerSettings(**env)
    producer = FakeProducer(fail_delivery=fail_delivery, buffer_errors=buffer_errors)
    holder: dict[str, Scorer] = {}
    consumer = FakeConsumer(batches, on_empty=lambda: holder["s"].stop())  # noqa: PLW0108 (bound later)
    scorer = Scorer(
        settings,
        KafkaSettings(),
        core=ScoringCore(FeatureEngine()),
        client=ModelClient("http://model", KEY, 1.0, transport=model_transport(fail_first)),
        consumer=consumer,  # type: ignore[arg-type]
        producer=producer,  # type: ignore[arg-type]
    )
    holder["s"] = scorer
    return scorer, consumer, producer


# ------------------------------------------------------------------ scorer


class TestScorerLoop:
    def test_scores_routes_dead_letters_and_stores_offsets(self) -> None:
        batch = [
            FakeMsg(TOPIC_RAW.name, raw_event(1), offset=10),
            FakeMsg(TOPIC_RAW.name, b"not json", offset=11),
            FakeMsg(TOPIC_RAW.name, raw_event(2), offset=12, partition=3),
            FakeMsg(TOPIC_RAW.name, None, offset=0, error=KafkaError(KafkaError._PARTITION_EOF)),
        ]
        scorer, consumer, producer = make_scorer([batch])
        scorer.run()
        assert consumer.subscribed == [TOPIC_RAW.name]
        assert producer.topics() == [TOPIC_SCORED.name, TOPIC_SCORED.name, TOPIC_DLQ.name]
        scored = [ScoredEvent.model_validate_json(v) for t, _, v in producer.sent[:2]]
        assert [s.tx_id for s in scored] == [f"{1:032x}", f"{2:032x}"]
        assert all(k == CARD.encode() for _, k, _ in producer.sent[:2])
        assert consumer.stored == [[(TOPIC_RAW.name, 0, 12), (TOPIC_RAW.name, 3, 13)]]
        assert consumer.closed

    def test_model_outage_pauses_then_resumes(self) -> None:
        scorer, consumer, producer = make_scorer(
            [[FakeMsg(TOPIC_RAW.name, raw_event(1))]], fail_first=2
        )
        scorer.run()
        assert len(consumer.paused) == 1
        assert len(consumer.resumed) == 1
        assert consumer.polls >= 1  # group membership kept alive while paused
        assert producer.topics() == [TOPIC_SCORED.name]
        assert consumer.stored

    def test_delivery_failure_does_not_store_offsets(self) -> None:
        scorer, consumer, _ = make_scorer(
            [[FakeMsg(TOPIC_RAW.name, raw_event(1))]], fail_delivery=True
        )
        with pytest.raises(RuntimeError, match="delivery failed"):
            scorer.run()
        assert consumer.stored == []
        assert consumer.closed

    def test_only_errors_is_a_no_op(self) -> None:
        err = FakeMsg(TOPIC_RAW.name, None, error=KafkaError(KafkaError._TRANSPORT))
        scorer, consumer, producer = make_scorer([[err]])
        scorer.run()
        assert producer.sent == []
        assert consumer.stored == []

    def test_revoke_drops_card_state(self) -> None:
        scorer, consumer, _ = make_scorer([[FakeMsg(TOPIC_RAW.name, raw_event(1), partition=2)]])
        scorer.run()
        assert scorer.core.engine.card_count == 1
        consumer.callbacks["on_revoke"](consumer, [TopicPartition(TOPIC_RAW.name, 2)])
        assert scorer.core.engine.card_count == 0

    @pytest.mark.parametrize("max_inflight", [1, 2, 4])
    def test_pipelined_batches_keep_order_and_store_offsets_in_order(
        self, max_inflight: int
    ) -> None:
        batches = [
            [FakeMsg(TOPIC_RAW.name, raw_event(n), offset=n, partition=n % 2)] for n in range(1, 9)
        ]
        scorer, consumer, producer = make_scorer(
            batches, max_inflight=max_inflight, buffer_errors=1
        )
        scorer.run()
        ids = [ScoredEvent.model_validate_json(v).tx_id for _, _, v in producer.sent]
        assert ids == [f"{n:032x}" for n in range(1, 9)]
        flat = [o for batch in consumer.stored for o in batch]
        assert flat == [(TOPIC_RAW.name, n % 2, n + 1) for n in range(1, 9)]

    def test_revoke_finishes_inflight_work_first(self) -> None:
        scorer, consumer, producer = make_scorer([])
        scorer.consumer.subscribe([TOPIC_RAW.name], on_revoke=scorer._on_revoke)
        scorer.submit([FakeMsg(TOPIC_RAW.name, raw_event(1), partition=2, offset=5)])  # type: ignore[list-item]
        consumer.callbacks["on_revoke"](consumer, [TopicPartition(TOPIC_RAW.name, 2)])
        assert producer.topics() == [TOPIC_SCORED.name]
        assert consumer.stored == [[(TOPIC_RAW.name, 2, 6)]]


# ------------------------------------------------------------------ producer


@pytest.fixture(scope="module")
def replay(tmp_path_factory: pytest.TempPathFactory) -> Replay:
    root = tmp_path_factory.mktemp("loops")
    ingest(write_kaggle_zip(root / "a.zip", n_cards=4), root / "data")
    return Replay.load(root / "data")


def run_producer(replay: Replay, fake: FakeProducer, **env: Any) -> int:
    settings = ProducerSettings(rate_tps=10_000, label_delay_s=0, **env)
    prod = TransactionProducer(settings, KafkaSettings(), replay=replay, producer=fake)  # type: ignore[arg-type]
    return prod.run()


class TestProducerLoop:
    def test_sends_transactions_then_labels(self, replay: Replay) -> None:
        fake = FakeProducer(buffer_errors=2)
        assert run_producer(replay, fake, max_events=30) == 0
        raw = [orjson.loads(v) for t, _, v in fake.sent if t == TOPIC_RAW.name]
        labels = [
            LabelEvent.model_validate_json(v) for t, _, v in fake.sent if t == TOPIC_LABELS.name
        ]
        assert len(raw) == 30
        assert all(r["produced_at_ns"] > 0 for r in raw)
        assert [lb.tx_id for lb in labels] == [r["tx_id"] for r in raw]
        assert all(k == r["card_id"].encode() for (t, k, _), r in zip(
            [s for s in fake.sent if s[0] == TOPIC_RAW.name], raw, strict=True))  # fmt: skip
        assert fake.polls >= 2  # BufferError -> poll -> retry

    def test_stop_ends_run(self, replay: Replay) -> None:
        fake = FakeProducer()
        settings = ProducerSettings(rate_tps=10_000, label_delay_s=0, loop=True)
        prod = TransactionProducer(settings, KafkaSettings(), replay=replay, producer=fake)  # type: ignore[arg-type]
        prod.stop()
        assert prod.run() == 0
        assert fake.sent == []


# ------------------------------------------------------------------ dashboard consumer


def scored_event(n: int, flagged: bool) -> ScoredEvent:
    return ScoredEvent(
        tx_id=f"{n:032x}", card_id=CARD, merchant="Shop", category="travel", amount=10.0,
        event_time=datetime(2020, 1, 1, tzinfo=UTC), score=0.9 if flagged else 0.1,
        is_flagged=flagged, threshold=0.5, model_version="fraud_xgb-abcdefabcdef",
        produced_at_ns=0, scored_at_ns=5_000_000,
    )  # fmt: skip


def label_event(n: int, is_fraud: bool) -> bytes:
    return LabelEvent(
        tx_id=f"{n:032x}", card_id=CARD, is_fraud=is_fraud,
        labelled_at=datetime(2020, 1, 1, tzinfo=UTC),
    ).model_dump_json().encode()  # fmt: skip


class TestDashboardConsumer:
    def test_handles_scored_labels_and_bad_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inserted: list[list[str]] = []
        labelled: list[list[tuple[str, bool]]] = []
        monkeypatch.setattr(
            dash_consumer, "insert_alerts", lambda _c, ev: inserted.append([e.tx_id for e in ev])
        )
        monkeypatch.setattr(dash_consumer, "apply_labels", lambda _c, lb: labelled.append(list(lb)))

        @contextmanager
        def fake_connect(**_kw: Any) -> Iterator[object]:
            yield object()

        monkeypatch.setattr(psycopg, "connect", fake_connect)
        batch = [
            FakeMsg(TOPIC_SCORED.name, scored_event(1, True).model_dump_json().encode(), offset=4),
            FakeMsg(TOPIC_SCORED.name, scored_event(2, False).model_dump_json().encode(), offset=5),
            FakeMsg(TOPIC_LABELS.name, label_event(1, True), offset=7),
            FakeMsg(TOPIC_LABELS.name, label_event(2, False), offset=8),  # not flagged: no row
            FakeMsg(TOPIC_LABELS.name, label_event(3, True), offset=9),  # unseen: still tried
            FakeMsg(TOPIC_SCORED.name, b"{}", offset=6),
            FakeMsg(TOPIC_SCORED.name, None, error=KafkaError(KafkaError._TRANSPORT)),
        ]
        stats = LiveStats(60)
        holder: dict[str, DashboardConsumer] = {}
        fake = FakeConsumer([batch], on_empty=lambda: holder["c"].stop())  # noqa: PLW0108 (bound later)
        dc = DashboardConsumer(
            stats, DashboardSettings(), KafkaSettings(), PG,
            consumer=fake,  # type: ignore[arg-type]
        )  # fmt: skip
        holder["c"] = dc
        dc.run()
        assert fake.subscribed == [TOPIC_SCORED.name, TOPIC_LABELS.name]
        assert inserted == [[f"{1:032x}", f"{2:032x}"]]
        assert labelled == [[(f"{1:032x}", True), (f"{3:032x}", True)]]
        assert dc.invalid_messages == 1
        assert fake.stored == [[(TOPIC_LABELS.name, 0, 10), (TOPIC_SCORED.name, 0, 7)]]
        snap = stats.snapshot()
        assert snap.as_dict()["tp"] == 1
        assert not dc.healthy.is_set()
        assert fake.closed

    def test_retries_until_database_is_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dash_consumer, "RECONNECT_MIN_S", 0.001)
        attempts = {"n": 0}
        holder: dict[str, DashboardConsumer] = {}

        @contextmanager
        def flaky(**_kw: Any) -> Iterator[object]:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise psycopg.OperationalError("db down")
            assert not holder["c"].healthy.is_set() or attempts["n"] == 3
            yield object()

        monkeypatch.setattr(psycopg, "connect", flaky)
        fake = FakeConsumer([], on_empty=lambda: holder["c"].stop())  # noqa: PLW0108 (bound later)
        dc = DashboardConsumer(
            LiveStats(60), DashboardSettings(), KafkaSettings(), PG,
            consumer=fake,  # type: ignore[arg-type]
        )  # fmt: skip
        holder["c"] = dc
        dc.run()
        assert attempts["n"] == 3
        assert dc.reconnects == 2
        assert fake.closed
        assert not dc.healthy.is_set()

    def test_failed_write_rewinds_without_counting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dash_consumer, "RECONNECT_MIN_S", 0.001)
        calls = {"n": 0}

        def insert(_c: object, events: list[ScoredEvent]) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise psycopg.OperationalError("connection lost mid-batch")
            return len(events)

        monkeypatch.setattr(dash_consumer, "insert_alerts", insert)
        monkeypatch.setattr(dash_consumer, "apply_labels", lambda _c, _lb: 0)

        @contextmanager
        def ok(**_kw: Any) -> Iterator[object]:
            yield object()

        monkeypatch.setattr(psycopg, "connect", ok)
        msg = FakeMsg(TOPIC_SCORED.name, scored_event(1, True).model_dump_json().encode(),
                      partition=2, offset=40)  # fmt: skip
        stats = LiveStats(60)
        holder: dict[str, DashboardConsumer] = {}
        # The real consumer re-delivers after a seek; the fake hands the batch out twice.
        fake = FakeConsumer([[msg], [msg]], on_empty=lambda: holder["c"].stop())  # noqa: PLW0108
        dc = DashboardConsumer(
            stats, DashboardSettings(), KafkaSettings(), PG,
            consumer=fake,  # type: ignore[arg-type]
        )  # fmt: skip
        holder["c"] = dc
        dc.run()
        assert fake.seeks == [(TOPIC_SCORED.name, 2, 40)]
        assert stats.scored_total == 1  # counted once, after the successful write
        assert fake.stored == [[(TOPIC_SCORED.name, 2, 41)]]

    def test_early_label_is_attached_when_flagged_score_arrives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        labelled: list[list[tuple[str, bool]]] = []
        monkeypatch.setattr(dash_consumer, "insert_alerts", lambda _c, _e: 0)
        monkeypatch.setattr(dash_consumer, "apply_labels", lambda _c, lb: labelled.append(list(lb)))

        @contextmanager
        def ok(**_kw: Any) -> Iterator[object]:
            yield object()

        monkeypatch.setattr(psycopg, "connect", ok)
        stats = LiveStats(60)
        holder: dict[str, DashboardConsumer] = {}
        batches = [
            [FakeMsg(TOPIC_LABELS.name, label_event(7, True), offset=1)],
            [FakeMsg(TOPIC_SCORED.name, scored_event(7, True).model_dump_json().encode())],
        ]
        fake = FakeConsumer(batches, on_empty=lambda: holder["c"].stop())  # noqa: PLW0108
        dc = DashboardConsumer(
            stats, DashboardSettings(), KafkaSettings(), PG,
            consumer=fake,  # type: ignore[arg-type]
        )  # fmt: skip
        holder["c"] = dc
        dc.run()
        tx = f"{7:032x}"
        assert labelled == [[(tx, True)], [(tx, True)]]
        assert stats.snapshot().tp == 1


def test_stats_ignore_repeated_deliveries() -> None:
    stats = LiveStats(60)
    event = scored_event(1, True)
    assert stats.record_scored(event) is True
    assert stats.record_scored(event) is False
    assert stats.record_label(event.tx_id, True) is True
    assert stats.record_label(event.tx_id, True) is None
    snap = stats.snapshot()
    assert (snap.scored_total, snap.flagged_total, snap.labelled_total, snap.tp) == (1, 1, 1, 1)
    assert stats.duplicates == 2
    assert stats.flag_status(event.tx_id) is True
    assert stats.flag_status("f" * 32) is None


# ------------------------------------------------------------------ restart recovery


class FakeReader:
    """Plays one partition's log back from the requested offset."""

    def __init__(self, log: dict[tuple[str, int], list[FakeMsg]], low: int = 0) -> None:
        self.log = log
        self.low = low
        self.assigned: list[tuple[str, int, int]] = []
        self._queue: list[FakeMsg] = []
        self.closed = False

    def get_watermark_offsets(self, tp: TopicPartition, timeout: float = 0) -> tuple[int, int]:
        assert timeout >= 0
        msgs = self.log.get((tp.topic, tp.partition), [])
        return self.low, (msgs[-1].offset() + 1 if msgs else self.low)

    def assign(self, parts: list[TopicPartition]) -> None:
        for tp in parts:
            self.assigned.append((tp.topic, tp.partition, tp.offset))
            self._queue += [
                m for m in self.log[(tp.topic, tp.partition)] if m.offset() >= tp.offset
            ]

    def consume(self, num_messages: int = 1, timeout: float = 0) -> list[FakeMsg]:
        assert timeout >= 0
        out, self._queue = self._queue[:num_messages], self._queue[num_messages:]
        return out

    def list_topics(self, topic: str, timeout: float = 0) -> Any:
        assert timeout >= 0
        parts = {p for (t, p) in self.log if t == topic}

        class Meta:
            topics: ClassVar[dict[str, Any]] = {
                topic: type("T", (), {"error": None, "partitions": dict.fromkeys(parts)})()
            }

        return Meta()

    def poll(self, timeout: float = 0) -> FakeMsg | None:
        got = self.consume(1, timeout)
        return got[0] if got else None

    def unassign(self) -> None:
        self._queue = []

    def close(self) -> None:
        self.closed = True


def raw_log(n: int, partition: int = 0, card: str = CARD) -> list[FakeMsg]:
    return [
        FakeMsg(TOPIC_RAW.name, raw_event(i, card), key=card.encode(), partition=partition,
                offset=i)
        for i in range(n)
    ]  # fmt: skip


class TestRestore:
    def test_replays_exactly_up_to_committed_offset(self) -> None:
        log_ = raw_log(10)
        log_[3] = FakeMsg(TOPIC_RAW.name, b"junk", partition=0, offset=3)  # went to the DLQ
        reader = FakeReader({(TOPIC_RAW.name, 0): log_})
        engine = FeatureEngine()
        n = restore_state(engine, reader, {(TOPIC_RAW.name, 0): 6}, max_per_partition=100)
        assert n == 5  # offsets 0..5 minus the junk one
        state = engine.state(CARD)
        assert state is not None
        assert state.count == 5

    def test_same_features_as_if_never_restarted(self) -> None:
        log_ = raw_log(8)
        reference = FeatureEngine()
        for m in log_:
            reference.process(TransactionEvent.model_validate_json(m.value() or b""))
        restored = FeatureEngine()
        restore_state(restored, FakeReader({(TOPIC_RAW.name, 0): log_}),
                      {(TOPIC_RAW.name, 0): 8}, max_per_partition=100)  # fmt: skip
        nxt = TransactionEvent.model_validate_json(raw_event(30))
        assert restored.process(nxt) == pytest.approx(reference.process(nxt), nan_ok=True)

    def test_cap_keeps_the_most_recent_events(self) -> None:
        reader = FakeReader({(TOPIC_RAW.name, 0): raw_log(10)})
        n = restore_state(FeatureEngine(), reader, {(TOPIC_RAW.name, 0): 10}, max_per_partition=4)
        assert n == 4
        assert reader.assigned == [(TOPIC_RAW.name, 0, 6)]

    def test_nothing_committed_means_nothing_to_do(self) -> None:
        reader = FakeReader({(TOPIC_RAW.name, 0): raw_log(3)})
        assert restore_state(FeatureEngine(), reader, {(TOPIC_RAW.name, 0): 0}, 10) == 0
        assert reader.assigned == []

    def test_scorer_restores_once_on_first_assignment(self) -> None:
        reader = FakeReader({(TOPIC_RAW.name, 2): raw_log(5, partition=2)})
        scorer, consumer, _ = make_scorer([])
        scorer._restore_reader = lambda: reader  # type: ignore[assignment,return-value]
        scorer._restored = False

        def committed(parts: list[TopicPartition], timeout: float = 0) -> list[TopicPartition]:
            assert timeout > 0
            return [TopicPartition(p.topic, p.partition, 5) for p in parts]

        consumer.committed = committed  # type: ignore[attr-defined]
        parts = [TopicPartition(TOPIC_RAW.name, 2)]
        scorer._on_assign(consumer, parts)  # type: ignore[arg-type]
        state = scorer.core.engine.state(CARD)
        assert state is not None
        assert state.count == 5
        assert reader.closed
        # A later assignment (rebalance) does not replay again.
        scorer._on_assign(consumer, parts)  # type: ignore[arg-type]
        assert state.count == 5


class TestTimeline:
    def test_next_run_resumes_right_after_what_kafka_already_has(self, replay: Replay) -> None:
        full = [e for e, _ in replay.iterate(loop=False)]
        assert replay.lap_of(None) == 0
        assert replay.lap_of(replay.first - timedelta(days=1)) == 0
        # Stopped part-way through lap 0: continue with the very next event, same ids.
        cut = full[len(full) // 3].event_time
        rest = [e for e, _ in replay.iterate(loop=False, after=cut)]
        assert rest[0].event_time > cut
        assert [e.tx_id for e in rest] == [e.tx_id for e in full if e.event_time > cut]
        # Stopped exactly at the end of lap 0: the next run is lap 1 (shifted, new ids).
        nxt = next(replay.iterate(loop=False, after=full[-1].event_time))[0]
        assert nxt.event_time == full[0].event_time + replay.span
        assert nxt.tx_id == _lap_id(full[0].tx_id, 1)
        # Part-way through lap 2.
        mid2 = full[10].event_time + replay.span * 2
        assert replay.lap_of(mid2) == 2
        assert next(replay.iterate(loop=False, after=mid2))[0].event_time > mid2

    def test_reads_newest_event_time_from_partition_tails(self) -> None:
        a = raw_log(3, partition=0)
        b = [FakeMsg(TOPIC_RAW.name, raw_event(59), partition=1, offset=0)]
        c = [FakeMsg(TOPIC_RAW.name, b"junk", partition=2, offset=0)]
        reader = FakeReader({(TOPIC_RAW.name, 0): a, (TOPIC_RAW.name, 1): b,
                             (TOPIC_RAW.name, 2): c})  # fmt: skip
        newest = read_last_event_time(reader, timeout_s=1)  # type: ignore[arg-type]
        assert newest == datetime(2020, 6, 21, 12, 59, tzinfo=UTC)
        assert read_last_event_time(FakeReader({}), timeout_s=1) is None  # type: ignore[arg-type]
