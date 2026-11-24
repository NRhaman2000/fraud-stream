"""End to end against the live stack (make up; make test-integration).

Kafka raw -> scorer -> model API -> scored / DLQ, and scored + labels -> dashboard ->
Postgres + REST API. Every test uses fresh random ids so it can run repeatedly while a
demo is streaming."""

import os
import secrets
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import httpx
import orjson
import pytest
from confluent_kafka import Consumer, Producer

from fraud_common.kafka import TOPIC_DLQ, TOPIC_LABELS, TOPIC_RAW, TOPIC_SCORED, client_config
from fraud_common.schemas import LabelEvent, ScoredEvent, TransactionEvent
from fraud_common.settings import KafkaSettings

pytestmark = pytest.mark.integration

MODEL_API = "http://localhost:8001"
DASHBOARD = "http://localhost:8000"
TIMEOUT_S = 30.0
E2E_EVENT_TIME = datetime(2018, 1, 1, tzinfo=UTC)


@pytest.fixture
def kafka() -> KafkaSettings:
    return KafkaSettings(bootstrap_servers="localhost:29092")


@pytest.fixture
def api_key() -> str:
    key = os.environ.get("IT_MODEL_API_KEY")
    if not key:
        pytest.fail("set IT_MODEL_API_KEY (make test-integration does this from .env)")
    return key


@pytest.fixture
def producer(kafka: KafkaSettings) -> Iterator[Producer]:
    p = Producer(client_config(kafka, "smoke") | {"acks": "all"})
    yield p
    p.flush(10)


@pytest.fixture
def tail(kafka: KafkaSettings) -> Iterator[Consumer]:
    """Consumer on scored + DLQ that only sees messages produced after it is assigned."""
    c = Consumer(
        client_config(kafka, "smoke")
        | {"group.id": f"it-e2e-{uuid.uuid4().hex[:8]}", "auto.offset.reset": "latest"}
    )
    c.subscribe([TOPIC_SCORED.name, TOPIC_DLQ.name])
    deadline = time.monotonic() + TIMEOUT_S
    while not c.assignment() and time.monotonic() < deadline:
        c.poll(0.2)
    assert c.assignment(), "e2e consumer never got partitions"
    # Pin every partition to its current end so nothing older is replayed.
    for tp in c.assignment():
        _, high = c.get_watermark_offsets(tp, timeout=10)
        tp.offset = high
        c.seek(tp)
    yield c
    c.close()


def tx_event(card: str, amount: float = 42.0) -> TransactionEvent:
    return TransactionEvent(
        tx_id=uuid.uuid4().hex,
        card_id=card,
        merchant="E2E Store",
        category="shopping_net",
        amount=amount,
        # Before the replayed period: the producer resumes after the newest event in Kafka,
        # so test events must never look newer than the demo stream.
        event_time=E2E_EVENT_TIME,
        merch_lat=40.7,
        merch_long=-74.0,
        produced_at_ns=time.time_ns(),
    )


def test_model_api_contract(api_key: str) -> None:
    assert httpx.get(f"{MODEL_API}/v1/model").status_code == 401
    info = httpx.get(f"{MODEL_API}/v1/model", headers={"X-API-Key": api_key}).json()
    assert info["model_version"].startswith("fraud_xgb-")
    assert 0 < info["threshold"] < 1
    assert httpx.get(f"{MODEL_API}/readyz").status_code == 200


def test_transaction_is_scored_and_poison_goes_to_dlq(producer: Producer, tail: Consumer) -> None:
    card = secrets.token_hex(8)  # unseen card: fresh feature state
    first, second = tx_event(card), tx_event(card, amount=999.0)
    key = card.encode()
    producer.produce(TOPIC_RAW.name, key=key, value=first.model_dump_json().encode())
    producer.produce(TOPIC_RAW.name, key=key, value=second.model_dump_json().encode())
    producer.produce(TOPIC_RAW.name, key=key, value=b'{"not": "a transaction"}')
    assert producer.flush(10) == 0

    scored: list[ScoredEvent] = []
    dlq: list[bytes] = []
    deadline = time.monotonic() + TIMEOUT_S
    while (len(scored) < 2 or not dlq) and time.monotonic() < deadline:
        m = tail.poll(0.5)
        if m is None or m.error() is not None or m.key() != key:
            continue
        if m.topic() == TOPIC_SCORED.name:
            scored.append(ScoredEvent.model_validate_json(m.value() or b""))
        else:
            dlq.append(m.value() or b"")

    assert [s.tx_id for s in scored] == [first.tx_id, second.tx_id]  # per-card order kept
    for s in scored:
        assert s.card_id == card
        assert 0.0 <= s.score <= 1.0
        assert s.is_flagged == (s.score >= s.threshold)
        assert s.model_version.startswith("fraud_xgb-")
        assert 0 < s.e2e_latency_ms < TIMEOUT_S * 1000
    assert len(dlq) == 1
    dead = orjson.loads(dlq[0])
    assert dead["stage"]
    assert dead["error"]
    assert dead["raw"]  # original bytes kept (base64) for replay


def test_flagged_event_becomes_reviewable_alert(producer: Producer) -> None:
    card = secrets.token_hex(8)
    event = ScoredEvent(
        tx_id=uuid.uuid4().hex, card_id=card, merchant="<script>x</script>",
        category="shopping_net", amount=1234.56, event_time=datetime.now(UTC), score=0.97,
        is_flagged=True, threshold=0.5, model_version="fraud_xgb-e2e000000000",
        produced_at_ns=time.time_ns(), scored_at_ns=time.time_ns(),
    )  # fmt: skip
    producer.produce(TOPIC_SCORED.name, key=card.encode(), value=event.model_dump_json().encode())
    assert producer.flush(10) == 0

    alert = None
    deadline = time.monotonic() + TIMEOUT_S
    while alert is None and time.monotonic() < deadline:
        page = httpx.get(f"{DASHBOARD}/api/alerts", params={"limit": 200}).json()
        alert = next((a for a in page["items"] if a["tx_id"] == event.tx_id), None)
        if alert is None:
            time.sleep(0.5)
    assert alert is not None, "flagged event never reached the alerts table"
    assert alert["merchant"] == "<script>x</script>"  # stored verbatim, rendered as text
    assert alert["status"] == "open"
    assert alert["true_label"] is None

    label = LabelEvent(
        tx_id=event.tx_id, card_id=card, is_fraud=True, labelled_at=datetime.now(UTC)
    )
    producer.produce(TOPIC_LABELS.name, key=card.encode(), value=label.model_dump_json().encode())
    assert producer.flush(10) == 0
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        page = httpx.get(f"{DASHBOARD}/api/alerts", params={"limit": 200}).json()
        alert = next(a for a in page["items"] if a["tx_id"] == event.tx_id)
        if alert["true_label"] is not None:
            break
        time.sleep(0.5)
    assert alert["true_label"] is True

    url = f"{DASHBOARD}/api/alerts/{event.tx_id}"
    resp = httpx.patch(url, json={"status": "confirmed_fraud"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "confirmed_fraud"
    assert httpx.patch(url, json={"status": "false_positive"}).status_code == 409


def test_dashboard_live_stats() -> None:
    assert httpx.get(f"{DASHBOARD}/readyz").status_code == 200
    stats = httpx.get(f"{DASHBOARD}/api/stats").json()
    assert {"tps", "p95_ms", "precision", "recall", "scored_total"} <= stats.keys()
    with httpx.stream("GET", f"{DASHBOARD}/api/stream", timeout=10) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = []
        for line in resp.iter_lines():
            lines.append(line)
            if line.startswith("event: stats"):
                break
    assert "event: stats" in lines
