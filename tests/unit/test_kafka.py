import pytest

from fraud_common.kafka import (
    ALL_TOPICS,
    TOPIC_DLQ,
    TOPIC_LABELS,
    TOPIC_RAW,
    TOPIC_SCORED,
    TopicSpec,
    client_config,
)
from fraud_common.settings import KafkaSettings


def test_topics_match_design_doc() -> None:
    assert [(t.name, t.partitions) for t in ALL_TOPICS] == [
        ("transactions.raw", 6),
        ("transactions.scored", 6),
        ("transactions.labels", 6),
        ("transactions.dlq", 1),
    ]
    assert (
        TOPIC_RAW.retention_ms
        == TOPIC_SCORED.retention_ms
        == TOPIC_LABELS.retention_ms
        == 86_400_000
    )
    assert TOPIC_DLQ.retention_ms == 7 * 86_400_000


def test_topic_names_unique() -> None:
    assert len({t.name for t in ALL_TOPICS}) == len(ALL_TOPICS)


def test_raw_and_scored_partitions_equal() -> None:
    # Keeps tx_id -> partition mapping identical end to end.
    assert TOPIC_RAW.partitions == TOPIC_SCORED.partitions == TOPIC_LABELS.partitions


@pytest.mark.parametrize(
    ("name", "partitions", "retention"), [("", 1, 1), ("x" * 250, 1, 1), ("t", 0, 1), ("t", 1, 0)]
)
def test_topic_spec_validation(name: str, partitions: int, retention: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        TopicSpec(name, partitions, retention)


def test_topic_config() -> None:
    assert TopicSpec("t", 1, 1000).topic_config() == {
        "retention.ms": "1000",
        "cleanup.policy": "delete",
    }


def test_client_config() -> None:
    cfg = client_config(KafkaSettings(), "scorer")
    assert cfg == {
        "bootstrap.servers": "localhost:29092",
        "client.id": "fraud-stream-scorer",
        "socket.nagle.disable": True,
    }
