"""Infrastructure bootstrap and verification.

    python -m fraud_common.admin ensure-topics   # create missing topics, verify the rest
    python -m fraud_common.admin check           # verify Kafka topics and the Postgres schema
    python -m fraud_common.admin ensure-db-role  # create/update the least-privilege app login

The decision logic (`plan_topics`, `diff_columns`) is pure and unit-tested; the thin I/O
wrappers around it are covered by the integration tests against the real stack.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import psycopg
from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient
from confluent_kafka.cimpl import NewTopic
from psycopg import sql

from fraud_common.kafka import ALL_TOPICS, TopicSpec, client_config
from fraud_common.logging_config import configure_logging
from fraud_common.settings import KafkaSettings, LoggingSettings, PostgresSettings

log = logging.getLogger(__name__)


class InfraError(RuntimeError):
    """The running infrastructure does not match what the code expects."""


# ---------------------------------------------------------------- Kafka


@dataclass(frozen=True, slots=True)
class TopicPlan:
    to_create: tuple[TopicSpec, ...] = ()
    mismatched: tuple[tuple[TopicSpec, int], ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.to_create and not self.mismatched


def plan_topics(specs: Sequence[TopicSpec], existing: Mapping[str, int]) -> TopicPlan:
    """Compare wanted topics with existing ones (name -> partition count)."""
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate topic names in specs")
    to_create = tuple(s for s in specs if s.name not in existing)
    mismatched = tuple(
        (s, existing[s.name])
        for s in specs
        if s.name in existing and existing[s.name] != s.partitions
    )
    return TopicPlan(to_create=to_create, mismatched=mismatched)


def _mismatch_message(plan: TopicPlan) -> str:
    return "; ".join(
        f"{spec.name} has {actual} partitions, expected {spec.partitions}"
        for spec, actual in plan.mismatched
    )


def existing_topics(admin: AdminClient, timeout_s: float) -> dict[str, int]:
    metadata = admin.list_topics(timeout=timeout_s)
    return {
        name: len(topic.partitions)
        for name, topic in metadata.topics.items()
        if topic.error is None and not name.startswith("__")
    }


def ensure_topics(
    admin: AdminClient, specs: Sequence[TopicSpec], timeout_s: float
) -> tuple[str, ...]:
    """Create missing topics. Never alters existing ones; a mismatch is an error so a
    wrong partition count cannot silently break consumer-group scaling."""
    plan = plan_topics(specs, existing_topics(admin, timeout_s))
    if plan.mismatched:
        raise InfraError(_mismatch_message(plan))
    if not plan.to_create:
        return ()
    futures = admin.create_topics(
        [
            NewTopic(s.name, num_partitions=s.partitions, config=s.topic_config())
            for s in plan.to_create
        ],
        operation_timeout=timeout_s,
        request_timeout=timeout_s,
    )
    created: list[str] = []
    for name, future in futures.items():
        try:
            future.result()
            created.append(name)
        except KafkaException as exc:
            # A concurrent creator winning the race is fine; anything else is not.
            if exc.args[0].name() != "TOPIC_ALREADY_EXISTS":
                raise InfraError(f"could not create topic {name}: {exc}") from exc
    return tuple(sorted(created))


def verify_topics(admin: AdminClient, specs: Sequence[TopicSpec], timeout_s: float) -> None:
    plan = plan_topics(specs, existing_topics(admin, timeout_s))
    problems = [f"missing topic {s.name}" for s in plan.to_create]
    if plan.mismatched:
        problems.append(_mismatch_message(plan))
    if problems:
        raise InfraError("; ".join(problems))


# ---------------------------------------------------------------- Postgres

EXPECTED_ALERT_COLUMNS: Final[Mapping[str, str]] = {
    "tx_id": "uuid",
    "card_id": "text",
    "merchant": "text",
    "category": "text",
    "score": "real",
    "amount": "numeric",
    "model_version": "text",
    "event_time": "timestamp with time zone",
    "flagged_at": "timestamp with time zone",
    "status": "USER-DEFINED",
    "true_label": "boolean",
    "reviewed_at": "timestamp with time zone",
}


def diff_columns(expected: Mapping[str, str], actual: Mapping[str, str]) -> list[str]:
    problems = [f"missing column {c}" for c in expected if c not in actual]
    problems += [
        f"column {c} is {actual[c]}, expected {t}"
        for c, t in expected.items()
        if c in actual and actual[c] != t
    ]
    problems += [f"unexpected column {c}" for c in actual if c not in expected]
    return problems


def verify_postgres(settings: PostgresSettings) -> None:
    with psycopg.connect(**settings.connect_kwargs()) as conn:  # type: ignore[arg-type]
        rows = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'alerts'"
        ).fetchall()
    problems = diff_columns(EXPECTED_ALERT_COLUMNS, {str(r[0]): str(r[1]) for r in rows})
    if problems:
        raise InfraError("alerts table: " + "; ".join(problems))


APP_GRANTS: Final = "SELECT, INSERT, UPDATE"  # never DELETE, never DDL


def ensure_app_role(settings: PostgresSettings) -> None:
    """Create (or re-key) the dashboard's login with only the rights it needs.

    Idempotent, so it runs on every start and also works on a database volume that was
    created before this role existed. Uses the admin login from POSTGRES_USER."""
    if settings.app_password is None:
        raise InfraError("POSTGRES_APP_PASSWORD is not set (see .env.example)")
    role = sql.Identifier(settings.app_user)
    password = sql.Literal(settings.app_password.get_secret_value())
    with psycopg.connect(**settings.connect_kwargs()) as conn:  # type: ignore[arg-type]
        exists = conn.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (settings.app_user,)
        ).fetchone()
        verb = "ALTER" if exists else "CREATE"
        conn.execute(
            sql.SQL(
                verb + " ROLE {} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}"
            ).format(role, password)
        )
        conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
            sql.Identifier(settings.db), role))  # fmt: skip
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role))
        conn.execute(sql.SQL("REVOKE ALL ON alerts FROM {}").format(role))
        conn.execute(sql.SQL(f"GRANT {APP_GRANTS} ON alerts TO {{}}").format(role))
    log.info("app role ready", extra={"role": settings.app_user, "action": verb.lower()})


# ---------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraud_common.admin")
    parser.add_argument("command", choices=["ensure-topics", "check", "ensure-db-role"])
    args = parser.parse_args(argv)

    configure_logging(LoggingSettings(), service="admin")
    if args.command == "ensure-db-role":
        try:
            ensure_app_role(PostgresSettings())
        except (InfraError, psycopg.Error) as exc:
            log.error("could not set up the app role", extra={"error": str(exc)})  # noqa: TRY400
            return 1
        return 0
    kafka = KafkaSettings()
    admin = AdminClient(client_config(kafka, "admin"))
    try:
        if args.command == "ensure-topics":
            created = ensure_topics(admin, ALL_TOPICS, kafka.admin_timeout_s)
            verify_topics(admin, ALL_TOPICS, kafka.admin_timeout_s)
            log.info("topics ready", extra={"topics_created": list(created)})
        else:
            verify_topics(admin, ALL_TOPICS, kafka.admin_timeout_s)
            verify_postgres(PostgresSettings())
            log.info("infrastructure check passed")
    except (InfraError, KafkaException, psycopg.Error) as exc:
        log.error("infrastructure check failed", extra={"error": str(exc)})  # noqa: TRY400
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
