import math
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from fraud_common.geo import haversine_km
from fraud_common.schemas import CardProfile, LabelEvent, ScoredEvent, TransactionEvent

GOOD: dict[str, Any] = {
    "tx_id": "0b6f4c1e2f6a4d4e9c393f1d5a2b7e10",
    "card_id": "0123456789abcdef",
    "merchant": "Kub and Mann",
    "category": "shopping_net",
    "amount": 149.62,
    "event_time": "2020-06-21T12:14:25Z",
    "merch_lat": 36.0,
    "merch_long": -81.2,
    "produced_at_ns": 1,
}


def test_valid_transaction_round_trips() -> None:
    event = TransactionEvent.model_validate(GOOD)
    assert event.schema_version == 2
    assert TransactionEvent.model_validate_json(event.model_dump_json()) == event


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tx_id", "not-hex"),
        ("tx_id", "0B6F4C1E2F6A4D4E9C393F1D5A2B7E10"),  # uppercase
        ("tx_id", "0b6f4c1e2f6a4d4e9c393f1d5a2b7e1"),  # 31 chars
        ("card_id", "0123456789abcde"),
        ("merchant", ""),
        ("merchant", "x" * 129),
        ("category", "Shopping"),
        ("category", "drop table;"),
        ("amount", 0),
        ("amount", -1),
        ("amount", 1_000_000.01),
        ("amount", math.inf),
        ("amount", math.nan),
        ("event_time", "2020-06-21T12:14:25"),  # naive datetime
        ("merch_lat", 90.1),
        ("merch_long", -180.1),
        ("produced_at_ns", -1),
        ("schema_version", 1),
    ],
)
def test_transaction_rejects(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        TransactionEvent.model_validate({**GOOD, field: value})


def test_transaction_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        TransactionEvent.model_validate({**GOOD, "is_fraud": 1})


def test_transaction_is_immutable() -> None:
    event = TransactionEvent.model_validate(GOOD)
    with pytest.raises(ValidationError):
        event.amount = 1.0  # type: ignore[misc]


def test_scored_event_latency() -> None:
    base = {k: v for k, v in GOOD.items() if k not in {"merch_lat", "merch_long"}}
    scored = ScoredEvent.model_validate(
        {
            **base,
            "score": 0.97,
            "is_flagged": True,
            "threshold": 0.6,
            "model_version": "m",
            "produced_at_ns": 1_000_000,
            "scored_at_ns": 8_500_000,
        }
    )
    assert scored.e2e_latency_ms == pytest.approx(7.5)
    clock_skew = scored.model_copy(update={"scored_at_ns": 0})
    assert clock_skew.e2e_latency_ms == 0


@pytest.mark.parametrize("score", [-0.01, 1.01, math.nan])
def test_scored_event_score_range(score: float) -> None:
    base = {k: v for k, v in GOOD.items() if k not in {"merch_lat", "merch_long"}}
    with pytest.raises(ValidationError):
        ScoredEvent.model_validate(
            {
                **base,
                "score": score,
                "is_flagged": True,
                "threshold": 0.5,
                "model_version": "m",
                "scored_at_ns": 2,
            }
        )


def test_label_and_profile() -> None:
    label = LabelEvent(
        tx_id=GOOD["tx_id"],
        card_id=GOOD["card_id"],
        is_fraud=True,
        labelled_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    assert label.is_fraud
    with pytest.raises(ValidationError):
        CardProfile(card_id=GOOD["card_id"], home_lat=0, home_long=0, city_pop=-1)


class TestHaversine:
    def test_zero(self) -> None:
        assert haversine_km(10, 20, 10, 20) == 0

    def test_known_distance_toronto_montreal(self) -> None:
        assert haversine_km(43.6532, -79.3832, 45.5019, -73.5674) == pytest.approx(504, abs=3)

    def test_symmetric(self) -> None:
        assert haversine_km(1, 2, 3, 4) == pytest.approx(haversine_km(3, 4, 1, 2))

    def test_antipodal_is_half_circumference(self) -> None:
        assert haversine_km(0, 0, 0, 180) == pytest.approx(math.pi * 6371.0088)

    def test_dateline(self) -> None:
        assert haversine_km(0, 179.9, 0, -179.9) == pytest.approx(22.2, abs=0.1)
