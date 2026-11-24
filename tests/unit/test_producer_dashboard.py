from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import pytest
from fastapi.testclient import TestClient

from fraud_common.alerts import (
    Alert,
    AlertAlreadyReviewedError,
    AlertNotFoundError,
    AlertStatus,
    PageCursor,
)
from fraud_common.features import FeatureEngine
from fraud_common.kafka import next_offsets
from fraud_common.schemas import ScoredEvent, TransactionEvent
from fraud_common.settings import DashboardSettings
from fraud_services.dashboard.app import create_app, sse
from fraud_services.dashboard.stats import LiveStats
from fraud_services.producer.main import LabelQueue, Pacer, Replay, _lap_id
from fraud_services.scorer.core import load_history, load_profiles
from fraud_training.ingest import ingest
from tests.fixtures.sparkov_like import write_kaggle_zip

# ------------------------------------------------------------------ producer


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def test_pacer_holds_rate() -> None:
    clock = FakeClock()
    pacer = Pacer(100, clock=clock, sleep=clock.sleep)
    for _ in range(101):
        pacer.wait()
    assert clock.t == pytest.approx(101.0)  # 100 intervals of 10 ms


def test_pacer_reanchors_after_stall() -> None:
    clock = FakeClock()
    pacer = Pacer(1000, clock=clock, sleep=clock.sleep)
    pacer.wait()
    clock.t += 5  # broker stalled for 5 s
    pacer.wait()
    clock.slept.clear()
    for _ in range(10):
        pacer.wait()
    assert sum(clock.slept) < 0.02  # no catch-up burst sleeping, no 5 s debt


def test_pacer_validation() -> None:
    with pytest.raises(ValueError, match="rate"):
        Pacer(0)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("prod")
    ingest(write_kaggle_zip(root / "a.zip", n_cards=8), root / "data")
    return root / "data"


def test_replay_is_the_test_period_in_order(data_dir: Path) -> None:
    replay = Replay.load(data_dir)
    events = list(replay.iterate(loop=False))
    times = [e.event_time for e, _ in events]
    assert times == sorted(times)
    assert min(times) >= datetime(2019, 7, 1, tzinfo=UTC)  # history comes from the scorer
    assert len({e.tx_id for e, _ in events}) == len(events)
    assert any(f for _, f in events)


def test_replay_loop_shifts_time_and_ids(data_dir: Path) -> None:
    replay = Replay.load(data_dir)
    n = replay.rows.num_rows
    it = replay.iterate(loop=True)
    first = [next(it)[0] for _ in range(n)]
    second = [next(it)[0] for _ in range(n)]
    assert first[0].tx_id != second[0].tx_id
    assert second[0].tx_id == _lap_id(first[0].tx_id, 1)
    assert second[0].event_time - first[0].event_time == replay.span
    assert second[0].event_time > first[-1].event_time  # per-card time keeps moving forward


def test_replay_requires_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        Replay.load(tmp_path)
    with pytest.raises(FileNotFoundError, match="make data"):
        load_profiles(tmp_path)
    with pytest.raises(FileNotFoundError, match="make data"):
        load_history(tmp_path, FeatureEngine())


def test_label_queue() -> None:
    q = LabelQueue(delay_s=10)
    ev = TransactionEvent(
        tx_id="a" * 32, card_id="b" * 16, merchant="m", category="travel", amount=1.0,
        event_time=datetime(2020, 1, 1, tzinfo=UTC), merch_lat=0, merch_long=0, produced_at_ns=0,
    )  # fmt: skip
    q.add(0.0, ev, True)
    q.add(5.0, ev.model_copy(update={"tx_id": "c" * 32}), False)
    assert q.due(9.9) == []
    assert [x.tx_id for x in q.due(10.0)] == ["a" * 32]
    assert len(q) == 1
    assert [x.is_fraud for x in q.due(0.0, flush_all=True)] == [False]


# ------------------------------------------------------------------ kafka helper


class Msg:
    def __init__(self, topic: str, part: int, off: int) -> None:
        self._t, self._p, self._o = topic, part, off

    def topic(self) -> str:
        return self._t

    def partition(self) -> int:
        return self._p

    def offset(self) -> int:
        return self._o


def test_next_offsets() -> None:
    tps = next_offsets([Msg("a", 0, 5), Msg("a", 0, 3), Msg("a", 1, 9), Msg("b", 0, 0)])
    assert [(t.topic, t.partition, t.offset) for t in tps] == [
        ("a", 0, 6),
        ("a", 1, 10),
        ("b", 0, 1),
    ]


# ------------------------------------------------------------------ dashboard stats


def scored(n: int, flagged: bool, latency_ms: float = 5.0) -> ScoredEvent:
    return ScoredEvent(
        tx_id=f"{n:032x}", card_id="b" * 16, merchant="<b>Shop</b>", category="travel",
        amount=10.0, event_time=datetime(2020, 1, 1, tzinfo=UTC), score=0.9 if flagged else 0.1,
        is_flagged=flagged, threshold=0.5, model_version="fraud_xgb-abcdefabcdef",
        produced_at_ns=0, scored_at_ns=int(latency_ms * 1e6),
    )  # fmt: skip


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_stats_window_and_percentiles() -> None:
    clock = Clock()
    stats = LiveStats(window_s=10, clock=clock)
    for i in range(100):
        stats.record_scored(scored(i, False, latency_ms=float(i + 1)))
    snap = stats.snapshot()
    assert snap.tps == 10.0
    assert snap.p50_ms == pytest.approx(50.5)
    assert snap.p95_ms == pytest.approx(95.05)
    clock.t = 11
    later = stats.snapshot()
    assert later.tps == 0
    assert later.p95_ms is None
    assert later.scored_total == 100  # totals are cumulative


def test_stats_join_labels_in_either_order() -> None:
    stats = LiveStats(window_s=60, clock=Clock())
    stats.record_scored(scored(1, True))  # TP: score then label
    assert stats.record_label(f"{1:032x}", True) is True
    assert stats.record_label(f"{2:032x}", False) is None  # FP: label first, then score
    stats.record_scored(scored(2, True))
    stats.record_scored(scored(3, False))  # FN
    stats.record_label(f"{3:032x}", True)
    stats.record_scored(scored(4, False))  # TN
    stats.record_label(f"{4:032x}", False)
    s = stats.snapshot()
    assert (s.tp, s.fp, s.fn, s.labelled_total) == (1, 1, 1, 4)
    assert s.precision == 0.5
    assert s.recall == 0.5
    assert s.flagged_total == 2


def test_stats_empty_and_validation() -> None:
    s = LiveStats(5, clock=Clock()).snapshot()
    assert s.precision is None
    assert s.recall is None
    assert s.tps == 0
    with pytest.raises(ValueError, match="window_s"):
        LiveStats(0)


def test_alert_feed_sequence() -> None:
    stats = LiveStats(60, clock=Clock())
    for i in range(5):
        stats.record_scored(scored(i, flagged=i % 2 == 0))
    notes = stats.alerts_after(0)
    assert [n.seq for n in notes] == [1, 2, 3]
    assert stats.alerts_after(2)[0].event.tx_id == f"{4:032x}"
    assert stats.last_seq == 3


def test_sse_format() -> None:
    assert sse("stats", {"a": 1}) == b'event: stats\ndata: {"a":1}\n\n'


# ------------------------------------------------------------------ dashboard API


def alert(tx: str, status: AlertStatus = AlertStatus.OPEN) -> Alert:
    now = datetime(2020, 1, 1, tzinfo=UTC)
    return Alert(
        tx_id=tx, card_id="b" * 16, merchant="m", category="travel", score=0.9, amount=1.0,
        model_version="v", event_time=now, flagged_at=now, status=status, true_label=None,
        reviewed_at=None if status is AlertStatus.OPEN else now,
    )  # fmt: skip


class FakeStore:
    def __init__(self) -> None:
        self.alerts = {f"{i:032x}": alert(f"{i:032x}") for i in range(3)}
        self.last_list: tuple[Any, ...] = ()
        self.up = True

    def list(
        self, status: AlertStatus | None, limit: int, before: PageCursor | None
    ) -> list[Alert]:
        self.last_list = (status, limit, before)
        return list(self.alerts.values())[:limit]

    def review(self, tx_id: str, decision: AlertStatus) -> Alert:
        if tx_id not in self.alerts:
            raise AlertNotFoundError(tx_id)
        if self.alerts[tx_id].status is not AlertStatus.OPEN:
            raise AlertAlreadyReviewedError(tx_id)
        self.alerts[tx_id] = alert(tx_id, decision)
        return self.alerts[tx_id]

    def ping(self) -> bool:
        return self.up


@pytest.fixture
def dash() -> Iterator[tuple[TestClient, FakeStore, dict[str, bool]]]:
    store, alive = FakeStore(), {"ok": True}
    stats = LiveStats(60)
    stats.record_scored(scored(1, True))
    app = create_app(DashboardSettings(), stats, store, lambda: alive["ok"])
    with TestClient(app, base_url="http://localhost") as c:
        yield c, store, alive


def test_page_and_static_assets(dash: Any) -> None:
    c, _, _ = dash
    page = c.get("/")
    assert page.status_code == 200
    assert "Alert queue" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"
    js = c.get("/static/app.js").text
    assert "innerHTML" not in js  # untrusted text only via textContent
    assert c.get("/static/style.css").status_code == 200


def test_stats_endpoint(dash: Any) -> None:
    c, _, _ = dash
    body = c.get("/api/stats").json()
    assert body["scored_total"] == 1
    assert body["flagged_total"] == 1


def test_list_alerts_and_paging(dash: Any) -> None:
    c, store, _ = dash
    body = c.get("/api/alerts", params={"limit": 2, "status": "open"}).json()
    assert len(body["items"]) == 2
    assert body["next_before_tx_id"] == f"{1:032x}"
    nxt = c.get("/api/alerts", params={
        "limit": 2, "before_flagged_at": body["next_before_flagged_at"],
        "before_tx_id": body["next_before_tx_id"],
    })  # fmt: skip
    assert nxt.status_code == 200
    assert store.last_list[2] == PageCursor.model_validate(
        {"flagged_at": body["next_before_flagged_at"], "tx_id": body["next_before_tx_id"]}
    )
    last = c.get("/api/alerts", params={"limit": 50}).json()
    assert last["next_before_tx_id"] is None


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 201},
        {"status": "maybe"},
        {"before_tx_id": "a" * 32},
        {"before_flagged_at": "2020-01-01T00:00:00Z"},
        {"before_flagged_at": "2020-01-01T00:00:00", "before_tx_id": "a" * 32},
        {"before_flagged_at": "2020-01-01T00:00:00Z", "before_tx_id": "nothex"},
    ],
)
def test_list_alerts_validation(dash: Any, params: dict[str, Any]) -> None:
    c, _, _ = dash
    resp = c.get("/api/alerts", params=params)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] in {"validation_error", "http_error"}


def test_review_flow(dash: Any) -> None:
    c, _, _ = dash
    tx = f"{0:032x}"
    ok = c.patch(f"/api/alerts/{tx}", json={"status": "confirmed_fraud"})
    assert ok.status_code == 200
    assert ok.json()["status"] == "confirmed_fraud"
    again = c.patch(f"/api/alerts/{tx}", json={"status": "false_positive"})
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "already_reviewed"
    missing = c.patch(f"/api/alerts/{'f' * 32}", json={"status": "false_positive"})
    assert missing.status_code == 404


@pytest.mark.parametrize(
    ("tx", "body"),
    [
        ("a" * 32, {"status": "open"}),
        ("a" * 32, {"status": "maybe"}),
        ("a" * 32, {"status": "confirmed_fraud", "extra": 1}),
        ("../../etc", {"status": "confirmed_fraud"}),
        ("A" * 32, {"status": "confirmed_fraud"}),
    ],
)
def test_review_validation(dash: Any, tx: str, body: dict[str, Any]) -> None:
    c, _, _ = dash
    assert c.patch(f"/api/alerts/{tx}", json=body).status_code in {404, 422}


def test_readiness(dash: Any) -> None:
    c, store, alive = dash
    assert c.get("/readyz").status_code == 200
    alive["ok"] = False
    assert c.get("/readyz").status_code == 503
    alive["ok"] = True
    store.up = False
    assert c.get("/readyz").json()["error"]["message"] == "database unreachable"
    assert c.get("/healthz").status_code == 200
    assert "dashboard_reviews_total" in c.get("/metrics").text


def test_alert_json_escapes_html() -> None:
    # The page renders via textContent; the JSON must carry the raw text unchanged.
    payload = orjson.loads(
        sse("alert", scored(1, True).model_dump(mode="json")).split(b"data: ")[1]
    )
    assert payload["merchant"] == "<b>Shop</b>"


def test_foreign_host_header_is_refused(dash: Any) -> None:
    c, _, _ = dash
    # A page on evil.example re-pointed at 127.0.0.1 still sends its own Host header.
    assert c.get("/api/alerts", headers={"Host": "evil.example"}).status_code == 400
    assert c.get("/api/alerts", headers={"Host": "dashboard:8000"}).status_code == 200
