"""Alert storage in PostgreSQL: the analyst review queue.

All SQL uses bound parameters (never string formatting), so card or merchant text can't
inject SQL. Writes are idempotent so at-least-once Kafka delivery never duplicates alerts.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Final

import psycopg
from psycopg.rows import dict_row
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from fraud_common.schemas import CardId, Probability, ScoredEvent, TxId

MAX_PAGE: Final = 200


class AlertStatus(StrEnum):
    OPEN = "open"
    CONFIRMED_FRAUD = "confirmed_fraud"
    FALSE_POSITIVE = "false_positive"


class AlertNotFoundError(LookupError):
    pass


class AlertAlreadyReviewedError(RuntimeError):
    pass


class PageCursor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    flagged_at: AwareDatetime
    tx_id: TxId

    @classmethod
    def after(cls, alert: Alert) -> PageCursor:
        return cls(flagged_at=alert.flagged_at, tx_id=alert.tx_id)


class Alert(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    tx_id: TxId
    card_id: CardId
    merchant: str
    category: str
    score: Probability
    amount: float = Field(ge=0)
    model_version: str
    event_time: AwareDatetime
    flagged_at: AwareDatetime
    status: AlertStatus
    true_label: bool | None
    reviewed_at: AwareDatetime | None


_COLUMNS: Final = (
    "tx_id, card_id, merchant, category, score, amount, model_version, event_time, "
    "flagged_at, status, true_label, reviewed_at"
)


def _to_alert(row: dict[str, Any]) -> Alert:
    row = dict(row)
    row["tx_id"] = row["tx_id"].hex
    row["amount"] = float(row["amount"])
    row["status"] = AlertStatus(row["status"])
    return Alert.model_validate(row)


def insert_alerts(conn: psycopg.Connection[Any], events: Sequence[ScoredEvent]) -> int:
    """Store flagged events; already-stored tx_ids are skipped. Returns rows inserted."""
    flagged = [e for e in events if e.is_flagged]
    if not flagged:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO alerts (tx_id, card_id, merchant, category, score, amount, "
            "model_version, event_time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (tx_id) DO NOTHING",
            [
                (
                    uuid.UUID(hex=e.tx_id),
                    e.card_id,
                    e.merchant,
                    e.category,
                    e.score,
                    round(e.amount, 2),
                    e.model_version,
                    e.event_time,
                )
                for e in flagged
            ],
            returning=False,
        )
        inserted = cur.rowcount
    conn.commit()
    return max(inserted, 0)


def list_alerts(
    conn: psycopg.Connection[Any],
    *,
    status: AlertStatus | None = None,
    limit: int = 50,
    before: PageCursor | None = None,
) -> list[Alert]:
    """Newest first. For the next page pass `PageCursor.after(last_row)`: keyset paging on
    (flagged_at, tx_id), because one insert batch shares a single flagged_at."""
    if not 1 <= limit <= MAX_PAGE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE}")
    params: dict[str, Any] = {
        "status": status.value if status else None,
        "limit": limit,
        "b_at": before.flagged_at if before else None,
        "b_tx": uuid.UUID(hex=before.tx_id) if before else None,
    }
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {_COLUMNS} FROM alerts "  # noqa: S608  (constant column list)
            "WHERE (%(status)s::alert_status IS NULL OR status = %(status)s::alert_status) "
            "AND (%(b_at)s::timestamptz IS NULL "
            "     OR (flagged_at, tx_id) < (%(b_at)s::timestamptz, %(b_tx)s::uuid)) "
            "ORDER BY flagged_at DESC, tx_id DESC LIMIT %(limit)s",
            params,
        )
        return [_to_alert(r) for r in cur.fetchall()]


def review_alert(conn: psycopg.Connection[Any], tx_id: str, status: AlertStatus) -> Alert:
    """Analyst decision. A reviewed alert cannot be reviewed again (409 in the API)."""
    if status is AlertStatus.OPEN:
        raise ValueError("a review must set confirmed_fraud or false_positive")
    key = uuid.UUID(hex=tx_id)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"UPDATE alerts SET status = %s, reviewed_at = now() "  # noqa: S608
            f"WHERE tx_id = %s AND status = 'open' RETURNING {_COLUMNS}",
            (status.value, key),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute("SELECT 1 FROM alerts WHERE tx_id = %s", (key,))
            exists = cur.fetchone() is not None
            conn.rollback()
            if exists:
                raise AlertAlreadyReviewedError(tx_id)
            raise AlertNotFoundError(tx_id)
    conn.commit()
    return _to_alert(row)


def apply_labels(conn: psycopg.Connection[Any], labels: Sequence[tuple[str, bool]]) -> int:
    """Attach delayed ground truth (tx_id, is_fraud) to existing alerts. Returns rows updated."""
    if not labels:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE alerts SET true_label = %s WHERE tx_id = %s",
            [(is_fraud, uuid.UUID(hex=tx_id)) for tx_id, is_fraud in labels],
        )
        updated = cur.rowcount
    conn.commit()
    return max(updated, 0)
