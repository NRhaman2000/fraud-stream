from datetime import UTC, datetime
from typing import Any

import httpx
import orjson
import pytest

from fraud_common.features import N_FEATURES, FeatureEngine
from fraud_common.schemas import ScoreBatchRequest, ScoreBatchResponse, ScoreItem, ScoreResult
from fraud_services.scorer.client import (
    Backoff,
    ModelClient,
    ModelRejectedError,
    ModelUnavailableError,
    score_with_retry,
)
from fraud_services.scorer.core import ScoringCore

CARD = "0123456789abcdef"


def raw(n: int = 1, card: str = CARD, **override: Any) -> bytes:
    event = {
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
    event.update(override)
    return orjson.dumps(event)


def response_for(req: ScoreBatchRequest, score: float = 0.9) -> ScoreBatchResponse:
    return ScoreBatchResponse(
        model_version="fraud_xgb-abcdefabcdef",
        threshold=0.5,
        results=[
            ScoreResult(tx_id=i.tx_id, score=score, is_flagged=score >= 0.5) for i in req.items
        ],
    )


class TestCore:
    def test_valid_batch(self) -> None:
        core = ScoringCore(FeatureEngine(), clock_ns=lambda: 9_000_000)
        batch = core.prepare([(CARD.encode(), raw(1)), (CARD.encode(), raw(2))])
        assert batch.dead == []
        assert batch.request is not None
        assert len(batch.request.items) == 2
        assert all(len(i.features) == N_FEATURES for i in batch.request.items)
        assert batch.request.items[0].features[9] is None  # NaN (no previous tx) -> null
        assert batch.request.items[1].features[9] == 60.0  # second tx: 60 s later
        scored = core.finish(batch.events, response_for(batch.request))
        assert [s.tx_id for s in scored] == [f"{1:032x}", f"{2:032x}"]
        assert scored[0].scored_at_ns == 9_000_000
        assert scored[0].e2e_latency_ms == pytest.approx(8.0)
        assert scored[0].model_version == "fraud_xgb-abcdefabcdef"

    @pytest.mark.parametrize(
        ("key", "value", "stage", "reason"),
        [
            (CARD.encode(), None, "parse", "empty"),
            (CARD.encode(), b"{not json", "validate", "json"),
            (CARD.encode(), raw(amount=-1), "validate", "amount"),
            (CARD.encode(), raw(schema_version=1), "validate", "schema_version"),
            (b"ffffffffffffffff", raw(), "validate", "key does not match"),
        ],
    )
    def test_bad_messages_go_to_dlq(
        self, key: bytes, value: bytes | None, stage: str, reason: str
    ) -> None:
        core = ScoringCore(FeatureEngine())
        batch = core.prepare([(key, value), (CARD.encode(), raw(5))])
        assert len(batch.dead) == 1
        dead = orjson.loads(batch.dead[0].value)
        assert dead["stage"] == stage
        assert reason in dead["error"].lower()
        assert batch.request is not None
        assert len(batch.request.items) == 1  # the good message still goes through
        assert core.engine.card_count == 1

    def test_all_bad_gives_no_request(self) -> None:
        batch = ScoringCore(FeatureEngine()).prepare([(None, b"x")])
        assert batch.request is None
        assert len(batch.dead) == 1

    def test_finish_checks_alignment(self) -> None:
        core = ScoringCore(FeatureEngine())
        batch = core.prepare([(CARD.encode(), raw(1)), (CARD.encode(), raw(2))])
        assert batch.request is not None
        short = response_for(ScoreBatchRequest(items=batch.request.items[:1]))
        with pytest.raises(ValueError, match="number"):
            core.finish(batch.events, short)
        swapped = response_for(ScoreBatchRequest(items=list(reversed(batch.request.items))))
        with pytest.raises(ValueError, match="order"):
            core.finish(batch.events, swapped)


def _client(handler: Any) -> ModelClient:
    return ModelClient("http://model", "k" * 40, 1.0, transport=httpx.MockTransport(handler))


REQ = ScoreBatchRequest(items=[ScoreItem(tx_id="a" * 32, features=[0.0] * N_FEATURES)])


class TestClient:
    def test_success_sends_key(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["key"] = request.headers["X-API-Key"]
            return httpx.Response(200, content=response_for(REQ).model_dump_json())

        assert _client(handler).score(REQ).results[0].tx_id == "a" * 32
        assert seen["key"] == "k" * 40

    @pytest.mark.parametrize("code", [429, 500, 503])
    def test_retryable_status(self, code: int) -> None:
        with pytest.raises(ModelUnavailableError):
            _client(lambda _r: httpx.Response(code)).score(REQ)

    @pytest.mark.parametrize("code", [400, 401, 413, 422])
    def test_non_retryable_status(self, code: int) -> None:
        with pytest.raises(ModelRejectedError):
            _client(lambda _r: httpx.Response(code, text="no")).score(REQ)

    def test_connection_error_is_retryable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(ModelUnavailableError, match="ConnectError"):
            _client(handler).score(REQ)


class TestRetry:
    def test_retries_until_success_and_keeps_alive(self) -> None:
        calls = {"n": 0, "alive": 0, "retries": 0}
        sleeps: list[float] = []

        def handler(_r: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 4:
                return httpx.Response(503)
            return httpx.Response(200, content=response_for(REQ).model_dump_json())

        backoff = Backoff(max_s=1.0, rand=lambda: 1.0)
        resp = score_with_retry(
            _client(handler),
            REQ,
            backoff,
            keep_alive=lambda: calls.__setitem__("alive", calls["alive"] + 1),
            sleep=sleeps.append,
            on_retry=lambda _e: calls.__setitem__("retries", calls["retries"] + 1),
        )
        assert resp.results
        assert sleeps == [0.05, 0.1, 0.2]
        assert calls["alive"] == calls["retries"] == 3
        assert backoff.attempt == 0  # reset after success

    def test_rejection_is_not_retried(self) -> None:
        with pytest.raises(ModelRejectedError):
            score_with_retry(
                _client(lambda _r: httpx.Response(401)), REQ, Backoff(1.0),
                keep_alive=lambda: None, sleep=lambda _s: None,
            )  # fmt: skip

    def test_backoff_caps_and_jitters(self) -> None:
        b = Backoff(max_s=0.3, rand=lambda: 1.0)
        assert [round(b.next_delay(), 3) for _ in range(5)] == [0.05, 0.1, 0.2, 0.3, 0.3]
        low = Backoff(max_s=0.3, rand=lambda: 0.0)
        assert low.next_delay() == 0.025  # floor: never a zero-delay hot loop

    def test_backoff_validation(self) -> None:
        with pytest.raises(ValueError, match="base_s"):
            Backoff(max_s=0.01, base_s=0.05)


def test_event_time_parsing_is_utc() -> None:
    core = ScoringCore(FeatureEngine())
    batch = core.prepare([(CARD.encode(), raw(1))])
    assert batch.events[0].event_time == datetime(2020, 6, 21, 12, 1, tzinfo=UTC)
