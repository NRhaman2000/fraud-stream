"""Wire formats for every Kafka event and the card profile store (design doc Revision 2).

Models are frozen and reject unknown fields, so a producer bug shows up as a DLQ entry
with a precise reason instead of a silently wrong score.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

SCHEMA_VERSION: Final = 2

TxId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$", description="32 lowercase hex chars")]
CardId = Annotated[str, Field(pattern=r"^[0-9a-f]{16}$", description="SHA-256 card token, 16 hex")]
Latitude = Annotated[float, Field(ge=-90.0, le=90.0, allow_inf_nan=False)]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]

MAX_AMOUNT: Final = 1_000_000.0


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=False)


class TransactionEvent(_Event):
    """`transactions.raw`, keyed by card_id."""

    schema_version: Literal[2] = SCHEMA_VERSION
    tx_id: TxId
    card_id: CardId
    merchant: str = Field(min_length=1, max_length=128)
    category: str = Field(pattern=r"^[a-z][a-z_]{0,63}$")
    amount: float = Field(gt=0.0, le=MAX_AMOUNT, allow_inf_nan=False)
    event_time: AwareDatetime
    merch_lat: Latitude
    merch_long: Longitude
    produced_at_ns: int = Field(ge=0)


class LabelEvent(_Event):
    """`transactions.labels`: ground truth arriving after a delay, like a chargeback."""

    schema_version: Literal[2] = SCHEMA_VERSION
    tx_id: TxId
    card_id: CardId
    is_fraud: bool
    labelled_at: AwareDatetime


class ScoredEvent(_Event):
    """`transactions.scored`, keyed by card_id."""

    schema_version: Literal[2] = SCHEMA_VERSION
    tx_id: TxId
    card_id: CardId
    merchant: str = Field(min_length=1, max_length=128)
    category: str = Field(pattern=r"^[a-z][a-z_]{0,63}$")
    amount: float = Field(gt=0.0, le=MAX_AMOUNT, allow_inf_nan=False)
    event_time: AwareDatetime
    score: Probability
    is_flagged: bool
    threshold: Probability
    model_version: str = Field(min_length=1, max_length=128)
    produced_at_ns: int = Field(ge=0)
    scored_at_ns: int = Field(ge=0)

    @property
    def e2e_latency_ms(self) -> float:
        return max(0, self.scored_at_ns - self.produced_at_ns) / 1e6


class CardProfile(_Event):
    """Static per-card data a bank would hold in its customer profile store."""

    card_id: CardId
    home_lat: Latitude
    home_long: Longitude
    city_pop: int = Field(ge=0)


# ---------------------------------------------------------------- model API contract

MAX_BATCH = 512


class ScoreItem(_Event):
    """One feature vector, ordered as `fraud_common.features.FEATURE_NAMES`; null = missing."""

    tx_id: TxId
    features: list[float | None]


class ScoreBatchRequest(_Event):
    items: list[ScoreItem] = Field(min_length=1, max_length=MAX_BATCH)


class ScoreResult(_Event):
    tx_id: TxId
    score: Probability
    is_flagged: bool


class ScoreBatchResponse(_Event):
    model_version: str
    threshold: Probability
    results: list[ScoreResult]
