"""Run against the live compose stack:  make up && make test-integration"""

import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient
from confluent_kafka.admin._config import ConfigResource
from pydantic import SecretStr

from fraud_common.admin import ensure_topics, existing_topics, verify_postgres, verify_topics
from fraud_common.kafka import ALL_TOPICS, TopicSpec, client_config
from fraud_common.settings import KafkaSettings, PostgresSettings

pytestmark = pytest.mark.integration

CARD = "0123456789abcdef"


@pytest.fixture
def kafka() -> KafkaSettings:
    return KafkaSettings(bootstrap_servers="localhost:29092")


@pytest.fixture
def pg() -> PostgresSettings:
    password = os.environ.get("IT_POSTGRES_PASSWORD")
    if not password:
        pytest.fail("set IT_POSTGRES_PASSWORD (make test-integration does this from .env)")
    return PostgresSettings(host="localhost", password=password)


@pytest.fixture
def conn(pg: PostgresSettings) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with psycopg.connect(**pg.connect_kwargs()) as c:  # type: ignore[arg-type]
        yield c
        c.rollback()


def test_all_topics_exist_with_right_partitions(kafka: KafkaSettings) -> None:
    verify_topics(AdminClient(client_config(kafka, "smoke")), ALL_TOPICS, 10)


def test_ensure_topics_is_idempotent(kafka: KafkaSettings) -> None:
    admin = AdminClient(client_config(kafka, "smoke"))
    assert ensure_topics(admin, ALL_TOPICS, 10) == ()


def test_ensure_creates_new_topic_with_retention(kafka: KafkaSettings) -> None:
    admin = AdminClient(client_config(kafka, "smoke"))
    spec = TopicSpec(f"it.{uuid.uuid4().hex[:8]}", 2, 60_000)
    try:
        assert ensure_topics(admin, [spec], 10) == (spec.name,)
        assert existing_topics(admin, 10)[spec.name] == 2
        res = ConfigResource("topic", spec.name)
        cfg = admin.describe_configs([res])[res].result()
        assert cfg["retention.ms"].value == "60000"
    finally:
        for f in admin.delete_topics([spec.name], operation_timeout=10).values():
            f.result()


def test_auto_topic_creation_disabled(kafka: KafkaSettings) -> None:
    name = f"it.auto.{uuid.uuid4().hex[:8]}"
    p = Producer(client_config(kafka, "smoke") | {"message.timeout.ms": "3000"})
    p.produce(name, b"x")
    p.flush(5)
    assert name not in existing_topics(AdminClient(client_config(kafka, "smoke")), 10)


def test_postgres_schema(pg: PostgresSettings) -> None:
    verify_postgres(pg)


def test_alert_insert_is_idempotent(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    tx = uuid.uuid4()
    sql = (
        "INSERT INTO alerts (tx_id, card_id, merchant, category, score, amount, model_version, "
        "event_time) VALUES (%s, '0123456789abcdef', 'm', 'travel', 0.9, 10.5, 'v1', now()) "
        "ON CONFLICT (tx_id) DO NOTHING"
    )
    assert conn.execute(sql, (tx,)).rowcount == 1
    assert conn.execute(sql, (tx,)).rowcount == 0


@pytest.mark.parametrize(
    ("score", "amount", "status", "reviewed", "card_id", "category"),
    [
        (1.5, 1, "open", False, CARD, "travel"),  # score > 1
        (-0.1, 1, "open", False, CARD, "travel"),  # score < 0
        (0.5, -1, "open", False, CARD, "travel"),  # negative amount
        (0.5, 1, "confirmed_fraud", False, CARD, "travel"),  # reviewed without timestamp
        (0.5, 1, "open", True, CARD, "travel"),  # open but has reviewed_at
        (0.5, 1, "maybe", False, CARD, "travel"),
        (0.5, 1, "open", False, "NOT-A-TOKEN", "travel"),
        (0.5, 1, "open", False, CARD, "Bad Category"),  # not in enum
    ],
)
def test_alert_constraints(
    conn: psycopg.Connection[tuple[object, ...]],
    score: float,
    amount: float,
    status: str,
    reviewed: bool,
    card_id: str,
    category: str,
) -> None:
    with pytest.raises(psycopg.Error):
        conn.execute(
            "INSERT INTO alerts (tx_id, card_id, merchant, category, score, amount, "
            "model_version, event_time, status, reviewed_at) "
            "VALUES (%s, %s, 'm', %s, %s, %s, 'v1', now(), %s, CASE WHEN %s THEN now() END)",
            (uuid.uuid4(), card_id, category, score, amount, status, reviewed),
        )


FORBIDDEN_FOR_APP = (
    "DELETE FROM alerts",
    "TRUNCATE alerts",
    "DROP TABLE alerts",
    "CREATE TABLE it_should_fail (x int)",
    "ALTER TABLE alerts ADD COLUMN x int",
)


def test_app_login_is_least_privilege(pg: PostgresSettings) -> None:
    """The dashboard's login can read, add and update alerts, and nothing else."""
    app_password = os.environ.get("IT_POSTGRES_APP_PASSWORD")
    if not app_password:
        pytest.fail("set IT_POSTGRES_APP_PASSWORD (make test-integration does this from .env)")
    app = pg.model_copy(update={"app_password": SecretStr(app_password)})
    with psycopg.connect(**app.app_connect_kwargs()) as c:  # type: ignore[arg-type]
        c.execute("SELECT count(*) FROM alerts").fetchone()
        for statement in FORBIDDEN_FOR_APP:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                c.execute(statement)
            c.rollback()
