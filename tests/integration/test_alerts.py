"""Alerts repository against the real Postgres:  make up && make test-integration"""

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from fraud_common.alerts import (
    MAX_PAGE,
    AlertAlreadyReviewedError,
    AlertNotFoundError,
    AlertStatus,
    PageCursor,
    apply_labels,
    insert_alerts,
    list_alerts,
    review_alert,
)
from fraud_common.schemas import ScoredEvent
from fraud_common.settings import PostgresSettings

pytestmark = pytest.mark.integration


@pytest.fixture
def conn() -> Iterator[psycopg.Connection[Any]]:
    password = os.environ.get("IT_POSTGRES_PASSWORD")
    if not password:
        pytest.fail("set IT_POSTGRES_PASSWORD (make test-integration does this from .env)")
    settings = PostgresSettings(host="localhost", password=password)
    with psycopg.connect(**settings.connect_kwargs()) as c:  # type: ignore[arg-type]
        c.execute("TRUNCATE alerts")
        c.commit()
        yield c
        c.rollback()
        c.execute("TRUNCATE alerts")
        c.commit()


def scored(flagged: bool = True, *, merchant: str = "Kub and Mann", **kw: Any) -> ScoredEvent:
    values: dict[str, Any] = {
        "tx_id": uuid.uuid4().hex,
        "card_id": "0123456789abcdef",
        "merchant": merchant,
        "category": "shopping_net",
        "amount": 149.625,
        "event_time": datetime(2020, 6, 21, 12, tzinfo=UTC),
        "score": 0.97,
        "is_flagged": flagged,
        "threshold": 0.6,
        "model_version": "fraud_xgb-abc123def456",
        "produced_at_ns": 1,
        "scored_at_ns": 2,
    }
    values.update(kw)
    return ScoredEvent(**values)


def test_insert_only_flagged_and_idempotent(conn: psycopg.Connection[Any]) -> None:
    events = [scored(), scored(), scored(flagged=False)]
    assert insert_alerts(conn, events) == 2
    assert insert_alerts(conn, events) == 0  # replay after a crash: no duplicates
    assert insert_alerts(conn, []) == 0
    rows = list_alerts(conn)
    assert {r.tx_id for r in rows} == {events[0].tx_id, events[1].tx_id}
    assert rows[0].amount == 149.62 or rows[0].amount == 149.63
    assert rows[0].status is AlertStatus.OPEN
    assert rows[0].true_label is None


def test_hostile_text_is_stored_verbatim(conn: psycopg.Connection[Any]) -> None:
    evil = "x'); DROP TABLE alerts; --"
    insert_alerts(conn, [scored(merchant=evil)])
    assert list_alerts(conn)[0].merchant == evil


def test_keyset_paging_covers_every_row_once(conn: psycopg.Connection[Any]) -> None:
    # One insert batch shares one flagged_at, which is exactly the tie case.
    events = [scored() for _ in range(7)]
    insert_alerts(conn, events)
    seen: list[str] = []
    cursor = None
    while True:
        page = list_alerts(conn, limit=3, before=cursor)
        if not page:
            break
        seen += [a.tx_id for a in page]
        cursor = PageCursor.after(page[-1])
    assert sorted(seen) == sorted(e.tx_id for e in events)
    assert len(seen) == len(set(seen))


def test_filter_by_status(conn: psycopg.Connection[Any]) -> None:
    a, b = scored(), scored()
    insert_alerts(conn, [a, b])
    review_alert(conn, a.tx_id, AlertStatus.FALSE_POSITIVE)
    assert [x.tx_id for x in list_alerts(conn, status=AlertStatus.OPEN)] == [b.tx_id]
    assert [x.tx_id for x in list_alerts(conn, status=AlertStatus.FALSE_POSITIVE)] == [a.tx_id]


@pytest.mark.parametrize("limit", [0, MAX_PAGE + 1])
def test_limit_bounds(conn: psycopg.Connection[Any], limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        list_alerts(conn, limit=limit)


def test_review_flow(conn: psycopg.Connection[Any]) -> None:
    event = scored()
    insert_alerts(conn, [event])
    reviewed = review_alert(conn, event.tx_id, AlertStatus.CONFIRMED_FRAUD)
    assert reviewed.status is AlertStatus.CONFIRMED_FRAUD
    assert reviewed.reviewed_at is not None
    assert reviewed.reviewed_at - datetime.now(UTC) < timedelta(minutes=1)
    with pytest.raises(AlertAlreadyReviewedError):
        review_alert(conn, event.tx_id, AlertStatus.FALSE_POSITIVE)
    assert list_alerts(conn)[0].status is AlertStatus.CONFIRMED_FRAUD  # first review stands


def test_review_unknown_and_invalid(conn: psycopg.Connection[Any]) -> None:
    with pytest.raises(AlertNotFoundError):
        review_alert(conn, uuid.uuid4().hex, AlertStatus.CONFIRMED_FRAUD)
    with pytest.raises(ValueError, match="review must set"):
        review_alert(conn, uuid.uuid4().hex, AlertStatus.OPEN)


def test_apply_labels(conn: psycopg.Connection[Any]) -> None:
    a, b = scored(), scored()
    insert_alerts(conn, [a, b])
    unknown = uuid.uuid4().hex  # label for a transaction that was never flagged
    assert apply_labels(conn, [(a.tx_id, True), (b.tx_id, False), (unknown, True)]) == 2
    labels = {x.tx_id: x.true_label for x in list_alerts(conn)}
    assert labels == {a.tx_id: True, b.tx_id: False}
    assert apply_labels(conn, []) == 0
