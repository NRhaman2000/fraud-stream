"""The one feature engine, shared by offline training and the live scorer.

Each card has a small state object. For every transaction the engine first computes the
feature vector from the card's history *before* this transaction, then folds the
transaction into the history. Training replays all historical events through exactly
this code, so the model never sees a feature computed differently from production.

Time is event time (when the purchase happened), never wall-clock time, so a replay at
5,000 tx/s produces the same features as the original two years of traffic.

Missing values are NaN (XGBoost learns a default direction for them): e.g. a card's first
transaction has no 'seconds since previous'.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final, NamedTuple

from fraud_common.geo import haversine_km
from fraud_common.schemas import CardProfile, TransactionEvent

# The 14 Sparkov categories. Order is part of the model contract: never reorder, only append.
CATEGORIES: Final[tuple[str, ...]] = (
    "entertainment",
    "food_dining",
    "gas_transport",
    "grocery_net",
    "grocery_pos",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel",
)
UNKNOWN_CATEGORY: Final = len(CATEGORIES)
_CATEGORY_INDEX: Final = {c: i for i, c in enumerate(CATEGORIES)}

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "amount",
    "log_amount",
    "category",
    "hour",
    "day_of_week",
    "log_city_pop",
    "km_from_home",
    "km_from_prev",
    "speed_kmh_from_prev",
    "secs_since_prev",
    "tx_count_1h",
    "tx_count_24h",
    "amount_sum_24h",
    "amount_zscore",
    "category_first_use",
    "prior_tx_count",
    # Added in model v2 (validation PR-AUC 0.972 -> 0.987):
    "amount_sum_1h",
    "tx_count_7d",
    "amount_sum_7d",
    "max_amount_24h",
    "distinct_categories_24h",
    "amount_to_card_mean",
    "amount_to_7d_mean",
    "hour_deviation",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("category",)
# XGBoost feature types, aligned with FEATURE_NAMES: "c" = categorical, "q" = quantitative.
FEATURE_TYPES: Final[tuple[str, ...]] = tuple(
    "c" if n in CATEGORICAL_FEATURES else "q" for n in FEATURE_NAMES
)
N_FEATURES: Final = len(FEATURE_NAMES)

HOUR_S: Final = 3600.0
DAY_S: Final = 86400.0
WEEK_S: Final = 7 * DAY_S
MAX_SPEED_KMH: Final = 20_000.0  # caps 'teleport' speeds so one outlier can't dominate
_NAN: Final = math.nan


def category_code(category: str) -> int:
    return _CATEGORY_INDEX.get(category, UNKNOWN_CATEGORY)


class _WindowStats(NamedTuple):
    count_1h: int
    sum_1h: float
    count_24h: int
    sum_24h: float
    max_24h: float
    distinct_categories_24h: int
    count_7d: int
    sum_7d: float


def _scan_window(window: deque[tuple[float, float, int]], ts: float) -> _WindowStats:
    """One pass over the card's last 7 days (entries are strictly before `ts`)."""
    count_1h = count_24h = 0
    sum_1h = sum_24h = sum_7d = max_24h = 0.0
    cats_24h: set[int] = set()
    for wts, wamt, wcat in window:
        sum_7d += wamt
        if wts > ts - DAY_S:
            count_24h += 1
            sum_24h += wamt
            max_24h = max(max_24h, wamt)
            cats_24h.add(wcat)
            if wts > ts - HOUR_S:
                count_1h += 1
                sum_1h += wamt
    return _WindowStats(
        count_1h, sum_1h, count_24h, sum_24h, max_24h, len(cats_24h), len(window), sum_7d
    )


def _movement(state: CardState, ts: float, lat: float, lon: float) -> tuple[float, float, float]:
    """(seconds, km, km/h) since the card's previous purchase; NaN for the first one."""
    if state.last_ts is None:
        return _NAN, _NAN, _NAN
    secs = ts - state.last_ts
    km = haversine_km(state.last_lat, state.last_long, lat, lon)
    return secs, km, min(MAX_SPEED_KMH, km / max(secs, 1.0) * HOUR_S)


def _habits(state: CardState, amount: float, hour_angle: float) -> tuple[float, float, float]:
    """(amount z-score, amount / card mean, hours from the card's usual time of day)."""
    if state.count >= 2:
        std = math.sqrt(state.m2 / (state.count - 1))
        zscore = (amount - state.mean) / std if std > 0 else 0.0
    else:
        zscore = _NAN
    if not state.count:
        return zscore, _NAN, _NAN
    usual = math.atan2(state.hour_sin, state.hour_cos)
    gap = abs(hour_angle - usual) % (2 * math.pi)
    return zscore, amount / state.mean, min(gap, 2 * math.pi - gap) * 24 / (2 * math.pi)


@dataclass(slots=True)
class CardState:
    """Everything the engine remembers about one card."""

    last_ts: float | None = None
    last_lat: float = 0.0
    last_long: float = 0.0
    # (ts, amount, category) for the last 7 days
    window: deque[tuple[float, float, int]] = field(default_factory=deque)
    count: int = 0  # Welford running stats of amount over the card's whole history
    mean: float = 0.0
    m2: float = 0.0
    categories: set[int] = field(default_factory=set)
    hour_sin: float = 0.0  # running sums of the purchase hour on the 24 h circle
    hour_cos: float = 0.0


class FeatureEngine:
    """Stateful, single-threaded. In the live system each scorer owns the cards of its
    Kafka partitions (the topic is keyed by card_id), so no locking is needed."""

    def __init__(self, profiles: Mapping[str, CardProfile] | None = None) -> None:
        self._profiles: Mapping[str, CardProfile] = profiles or {}
        self._cards: dict[str, CardState] = {}

    @property
    def card_count(self) -> int:
        return len(self._cards)

    def state(self, card_id: str) -> CardState | None:
        return self._cards.get(card_id)

    def forget(self, card_ids: Iterable[str]) -> int:
        """Drop state for cards this process no longer owns. Returns how many were held."""
        return sum(self._cards.pop(c, None) is not None for c in card_ids)

    def process(self, tx: TransactionEvent) -> list[float]:
        """Features for `tx` from history before it, then add `tx` to the history."""
        return self.process_raw(
            card_id=tx.card_id,
            ts=tx.event_time.timestamp(),
            amount=tx.amount,
            category=category_code(tx.category),
            merch_lat=tx.merch_lat,
            merch_long=tx.merch_long,
        )

    def process_raw(  # noqa: PLR0913  keyword-only hot path, avoids building objects
        self,
        *,
        card_id: str,
        ts: float,
        amount: float,
        category: int,
        merch_lat: float,
        merch_long: float,
    ) -> list[float]:
        """Fast path without Pydantic, used by training over ~1.85M rows. `ts` is UTC epoch
        seconds; `category` is a code from `category_code`."""
        if not (math.isfinite(ts) and math.isfinite(amount) and amount > 0):
            raise ValueError(f"invalid transaction: ts={ts!r} amount={amount!r}")
        state = self._cards.get(card_id)
        if state is None:
            state = self._cards[card_id] = CardState()

        # Event time per card must not go backwards; a late event is treated as happening
        # at the card's latest known time, so windows and gaps stay well defined.
        if state.last_ts is not None and ts < state.last_ts:
            ts = state.last_ts

        window = state.window
        while window and window[0][0] <= ts - WEEK_S:
            window.popleft()

        w = _scan_window(window, ts)

        secs_prev, km_prev, speed = _movement(state, ts, merch_lat, merch_long)
        hour_angle = 2 * math.pi * (ts % DAY_S) / DAY_S
        zscore, to_card_mean, hour_dev = _habits(state, amount, hour_angle)
        to_7d_mean = amount / (w.sum_7d / w.count_7d) if w.count_7d else _NAN

        profile = self._profiles.get(card_id)
        if profile is None:
            km_home = log_pop = _NAN
        else:
            km_home = haversine_km(profile.home_lat, profile.home_long, merch_lat, merch_long)
            log_pop = math.log1p(profile.city_pop)

        day_index = math.floor(ts / DAY_S)
        features = [
            amount,
            math.log1p(amount),
            float(category),
            float(math.floor((ts % DAY_S) / HOUR_S)),
            float((day_index + 3) % 7),  # 1970-01-01 was a Thursday; Monday = 0
            log_pop,
            km_home,
            km_prev,
            speed,
            secs_prev,
            float(w.count_1h),
            float(w.count_24h),
            w.sum_24h,
            zscore,
            0.0 if category in state.categories else 1.0,
            float(state.count),
            w.sum_1h,
            float(w.count_7d),
            w.sum_7d,
            w.max_24h,
            float(w.distinct_categories_24h),
            to_card_mean,
            to_7d_mean,
            hour_dev,
        ]

        # Fold this transaction into the card's history.
        window.append((ts, amount, category))
        state.last_ts = ts
        state.last_lat, state.last_long = merch_lat, merch_long
        state.count += 1
        delta = amount - state.mean
        state.mean += delta / state.count
        state.m2 += delta * (amount - state.mean)
        state.categories.add(category)
        state.hour_sin += math.sin(hour_angle)
        state.hour_cos += math.cos(hour_angle)
        return features
