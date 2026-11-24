"""Kafka topic definitions and client config. This module is the single source of truth
for topic names, partition counts and retention: topic creation, verification and every
producer/consumer read from here."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from confluent_kafka import TopicPartition

from fraud_common.settings import KafkaSettings

_HOUR_MS: Final = 60 * 60 * 1000


@dataclass(frozen=True, slots=True)
class TopicSpec:
    name: str
    partitions: int
    retention_ms: int

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 249:
            raise ValueError("topic name must be 1-249 characters")
        if self.partitions < 1:
            raise ValueError("partitions must be >= 1")
        if self.retention_ms < 1:
            raise ValueError("retention_ms must be >= 1")

    def topic_config(self) -> dict[str, str]:
        return {"retention.ms": str(self.retention_ms), "cleanup.policy": "delete"}


TOPIC_RAW: Final = TopicSpec("transactions.raw", partitions=6, retention_ms=24 * _HOUR_MS)
TOPIC_SCORED: Final = TopicSpec("transactions.scored", partitions=6, retention_ms=24 * _HOUR_MS)
TOPIC_LABELS: Final = TopicSpec("transactions.labels", partitions=6, retention_ms=24 * _HOUR_MS)
TOPIC_DLQ: Final = TopicSpec("transactions.dlq", partitions=1, retention_ms=7 * 24 * _HOUR_MS)

ALL_TOPICS: Final[tuple[TopicSpec, ...]] = (TOPIC_RAW, TOPIC_SCORED, TOPIC_LABELS, TOPIC_DLQ)

Role = Literal["admin", "producer", "scorer", "dashboard", "smoke"]


ClientConfig = dict[str, str | int | float | bool]


def client_config(settings: KafkaSettings, role: Role) -> ClientConfig:
    """Base librdkafka config shared by every client; callers add role-specific keys."""
    return {
        "bootstrap.servers": settings.bootstrap_servers,
        "client.id": f"{settings.client_id_prefix}-{role}",
        # Send small requests immediately instead of waiting to coalesce them (Nagle);
        # this is a latency-first pipeline and librdkafka already batches on its own.
        "socket.nagle.disable": True,
    }


class _Positioned(Protocol):
    def topic(self) -> str | None: ...
    def partition(self) -> int | None: ...
    def offset(self) -> int | None: ...


def next_offsets(messages: Sequence[_Positioned]) -> list[TopicPartition]:
    """Offsets to store after processing `messages`: highest offset + 1 per partition."""
    highest: dict[tuple[str, int], int] = {}
    for m in messages:
        topic, partition, offset = m.topic(), m.partition(), m.offset()
        if topic is None or partition is None or offset is None:
            raise ValueError("message has no position")
        key = (topic, partition)
        highest[key] = max(highest.get(key, -1), offset)
    return [TopicPartition(t, p, o + 1) for (t, p), o in sorted(highest.items())]
