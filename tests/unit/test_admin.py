from typing import Any
from unittest.mock import MagicMock

import pytest
from confluent_kafka import KafkaError, KafkaException

from fraud_common import admin
from fraud_common.admin import (
    EXPECTED_ALERT_COLUMNS,
    InfraError,
    TopicPlan,
    diff_columns,
    ensure_topics,
    plan_topics,
    verify_topics,
)
from fraud_common.kafka import ALL_TOPICS, TopicSpec

A = TopicSpec("a", 6, 1)
B = TopicSpec("b", 1, 1)


class TestPlanTopics:
    def test_nothing_exists(self) -> None:
        assert plan_topics([A, B], {}) == TopicPlan(to_create=(A, B))

    def test_all_exist_and_match(self) -> None:
        plan = plan_topics([A, B], {"a": 6, "b": 1, "other": 3})
        assert plan.ok

    def test_partition_mismatch(self) -> None:
        plan = plan_topics([A, B], {"a": 3})
        assert plan.to_create == (B,)
        assert plan.mismatched == ((A, 3),)
        assert not plan.ok

    def test_duplicate_specs_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            plan_topics([A, A], {})

    def test_empty_specs(self) -> None:
        assert plan_topics([], {"a": 1}).ok


class TestDiffColumns:
    def test_match(self) -> None:
        assert diff_columns(EXPECTED_ALERT_COLUMNS, dict(EXPECTED_ALERT_COLUMNS)) == []

    def test_missing_wrong_and_extra(self) -> None:
        actual = dict(EXPECTED_ALERT_COLUMNS)
        del actual["score"]
        actual["amount"] = "text"
        actual["surprise"] = "int"
        assert diff_columns(EXPECTED_ALERT_COLUMNS, actual) == [
            "missing column score",
            "column amount is text, expected numeric",
            "unexpected column surprise",
        ]

    def test_table_absent(self) -> None:
        assert len(diff_columns(EXPECTED_ALERT_COLUMNS, {})) == len(EXPECTED_ALERT_COLUMNS)


# ---- I/O wrappers with a fake AdminClient


def _fake_admin(
    existing: dict[str, int], create_errors: dict[str, KafkaException] | None = None
) -> Any:
    fake = MagicMock()
    topics = {}
    for name, n in existing.items():
        t = MagicMock()
        t.partitions = dict.fromkeys(range(n))
        t.error = None
        topics[name] = t
    internal = MagicMock(partitions={0: None}, error=None)
    topics["__consumer_offsets"] = internal
    fake.list_topics.return_value.topics = topics

    def create_topics(new_topics: list[Any], **_: Any) -> dict[str, Any]:
        result = {}
        for nt in new_topics:
            fut = MagicMock()
            err = (create_errors or {}).get(nt.topic)
            fut.result.side_effect = err
            result[nt.topic] = fut
        return result

    fake.create_topics.side_effect = create_topics
    return fake


def test_ensure_creates_missing_only() -> None:
    fake = _fake_admin({"a": 6})
    assert ensure_topics(fake, [A, B], 1.0) == ("b",)
    (created,), _ = fake.create_topics.call_args
    assert [t.topic for t in created] == ["b"]


def test_ensure_noop_when_ready() -> None:
    fake = _fake_admin({"a": 6, "b": 1})
    assert ensure_topics(fake, [A, B], 1.0) == ()
    fake.create_topics.assert_not_called()


def test_ensure_refuses_partition_mismatch() -> None:
    with pytest.raises(InfraError, match="a has 2 partitions, expected 6"):
        ensure_topics(_fake_admin({"a": 2}), [A], 1.0)


def test_ensure_tolerates_concurrent_create() -> None:
    race = KafkaException(KafkaError(KafkaError.TOPIC_ALREADY_EXISTS))
    assert ensure_topics(_fake_admin({}, {"a": race}), [A, B], 1.0) == ("b",)


def test_ensure_surfaces_other_errors() -> None:
    denied = KafkaException(KafkaError(KafkaError.TOPIC_AUTHORIZATION_FAILED))
    with pytest.raises(InfraError, match="could not create topic a"):
        ensure_topics(_fake_admin({}, {"a": denied}), [A], 1.0)


def test_verify_reports_missing_and_mismatch() -> None:
    with pytest.raises(InfraError, match=r"missing topic b; a has 1 partitions"):
        verify_topics(_fake_admin({"a": 1}), [A, B], 1.0)


def test_verify_ignores_internal_topics() -> None:
    verify_topics(_fake_admin({"a": 6}), [A], 1.0)


# ---- CLI exit codes


def test_cli_rejects_unknown_command() -> None:
    with pytest.raises(SystemExit):
        admin.main(["drop-everything"])


def test_cli_returns_1_on_infra_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admin, "AdminClient", lambda _cfg: _fake_admin({}))
    assert admin.main(["check"]) == 1


def test_cli_ensure_topics_success(monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict[str, int] = {}
    fake = _fake_admin(state)

    def create(new_topics: list[Any], **_: Any) -> dict[str, Any]:
        for nt in new_topics:
            t = MagicMock(partitions=dict.fromkeys(range(nt.num_partitions)), error=None)
            fake.list_topics.return_value.topics[nt.topic] = t
        return {nt.topic: MagicMock() for nt in new_topics}

    fake.create_topics.side_effect = create
    monkeypatch.setattr(admin, "AdminClient", lambda _cfg: fake)
    assert admin.main(["ensure-topics"]) == 0
    assert set(fake.list_topics.return_value.topics) >= {t.name for t in ALL_TOPICS}


def test_cli_check_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_admin({t.name: t.partitions for t in ALL_TOPICS})
    monkeypatch.setattr(admin, "AdminClient", lambda _cfg: fake)
    monkeypatch.setattr(admin, "verify_postgres", lambda _s: None)
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    assert admin.main(["check"]) == 0
