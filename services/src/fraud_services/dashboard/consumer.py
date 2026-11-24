"""Background thread: transactions.scored + transactions.labels -> LiveStats and Postgres."""

from __future__ import annotations

import logging
import threading
from typing import Any

import psycopg
from confluent_kafka import Consumer, KafkaError, TopicPartition
from pydantic import ValidationError

from fraud_common.alerts import apply_labels, insert_alerts
from fraud_common.kafka import TOPIC_LABELS, TOPIC_SCORED, client_config, next_offsets
from fraud_common.schemas import LabelEvent, ScoredEvent
from fraud_common.settings import DashboardSettings, KafkaSettings, PostgresSettings
from fraud_services.dashboard.stats import LiveStats

log = logging.getLogger(__name__)
BATCH = 500
RECONNECT_MIN_S = 0.5
RECONNECT_MAX_S = 10.0


class DashboardConsumer(threading.Thread):
    def __init__(
        self,
        stats: LiveStats,
        settings: DashboardSettings,
        kafka: KafkaSettings,
        postgres: PostgresSettings,
        *,
        consumer: Consumer | None = None,
    ) -> None:
        super().__init__(name="dashboard-consumer", daemon=True)
        self.stats = stats
        self._postgres = postgres
        self._consumer = consumer or Consumer(
            client_config(kafka, "dashboard")
            | {
                "group.id": settings.group_id,
                "auto.offset.reset": "latest",  # the dashboard shows live traffic only
                "enable.auto.commit": True,
                "enable.auto.offset.store": False,
            }
        )
        self._stop = threading.Event()
        self.healthy = threading.Event()
        self.invalid_messages = 0
        self.reconnects = 0

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """Consume until stopped. A database outage is survived: the failed batch is
        re-read after reconnecting, and readiness is false while disconnected."""
        self._consumer.subscribe([TOPIC_SCORED.name, TOPIC_LABELS.name])
        backoff = RECONNECT_MIN_S
        try:
            while not self._stop.is_set():
                try:
                    with psycopg.connect(**self._postgres.app_connect_kwargs()) as conn:  # type: ignore[arg-type]
                        self.healthy.set()
                        backoff = RECONNECT_MIN_S
                        log.info("dashboard consumer connected to postgres")
                        self._loop(conn)
                except psycopg.Error as exc:
                    self.healthy.clear()
                    self.reconnects += 1
                    log.warning("postgres unavailable; retrying",
                                extra={"error": str(exc)[:200], "retry_in_s": backoff})  # fmt: skip
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, RECONNECT_MAX_S)
        finally:
            self.healthy.clear()
            self._consumer.close()

    def _loop(self, conn: psycopg.Connection[Any]) -> None:
        while not self._stop.is_set():
            messages = self._consumer.consume(num_messages=BATCH, timeout=0.2)
            if not messages:
                continue
            try:
                self._handle(conn, messages)
            except psycopg.Error:
                self._rewind(messages)  # nothing of this batch is lost or double counted
                raise

    def _rewind(self, messages: list[Any]) -> None:
        first: dict[tuple[str, int], int] = {}
        for m in messages:
            if m.error() is None:
                key = (m.topic(), m.partition())
                first[key] = min(first.get(key, m.offset()), m.offset())
        for (topic, partition), offset in first.items():
            self._consumer.seek(TopicPartition(topic, partition, offset))

    def _handle(self, conn: psycopg.Connection[Any], messages: list[Any]) -> None:
        """Write to Postgres first, then update the live stats, so a database failure
        leaves the stats untouched and the batch can simply be read again."""
        scored: list[ScoredEvent] = []
        labels: list[LabelEvent] = []
        for m in messages:
            if m.error() is not None:
                if m.error().code() != KafkaError._PARTITION_EOF:
                    log.error("consumer error", extra={"error": str(m.error())})
                continue
            try:
                if m.topic() == TOPIC_SCORED.name:
                    scored.append(ScoredEvent.model_validate_json(m.value()))
                else:
                    labels.append(LabelEvent.model_validate_json(m.value()))
            except ValidationError:
                self.invalid_messages += 1
                log.warning("skipping invalid message", extra={"topic": m.topic()})

        # Only flagged transactions have a row. Labels for unknown or flagged ones are
        # applied (the row may exist from an earlier batch); a label that arrived before
        # its score is attached when the flagged score arrives.
        in_batch = {e.tx_id: e.is_flagged for e in scored}
        updates = [
            (lb.tx_id, lb.is_fraud)
            for lb in labels
            if in_batch.get(lb.tx_id, self.stats.flag_status(lb.tx_id)) is not False
        ]
        for event in scored:
            if event.is_flagged and (early := self.stats.pending_label(event.tx_id)) is not None:
                updates.append((event.tx_id, early))
        insert_alerts(conn, scored)
        apply_labels(conn, updates)

        for event in scored:
            self.stats.record_scored(event)
        for lb in labels:
            self.stats.record_label(lb.tx_id, lb.is_fraud)
        good = [m for m in messages if m.error() is None]
        if good:
            self._consumer.store_offsets(offsets=next_offsets(good))
