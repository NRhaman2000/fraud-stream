import math
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fraud_common.features import (
    CATEGORIES,
    FEATURE_NAMES,
    FEATURE_TYPES,
    MAX_SPEED_KMH,
    N_FEATURES,
    UNKNOWN_CATEGORY,
    FeatureEngine,
    category_code,
)
from fraud_common.geo import haversine_km
from fraud_common.schemas import CardProfile, TransactionEvent

CARD = "a" * 16
OTHER = "b" * 16
T0 = datetime(2020, 6, 1, 12, 0, tzinfo=UTC)  # a Monday
TORONTO = (43.6532, -79.3832)
MONTREAL = (45.5019, -73.5674)


def tx(
    minutes: float = 0,
    amount: float = 50.0,
    category: str = "grocery_pos",
    where: tuple[float, float] = TORONTO,
    card: str = CARD,
    n: int = 0,
) -> TransactionEvent:
    return TransactionEvent(
        tx_id=f"{n:032x}",
        card_id=card,
        merchant="Shop",
        category=category,
        amount=amount,
        event_time=T0 + timedelta(minutes=minutes),
        merch_lat=where[0],
        merch_long=where[1],
        produced_at_ns=0,
    )


def named(vec: list[float]) -> dict[str, float]:
    assert len(vec) == N_FEATURES
    return dict(zip(FEATURE_NAMES, vec, strict=True))


def test_feature_contract_is_stable() -> None:
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == len(FEATURE_TYPES) == 24
    assert FEATURE_TYPES[FEATURE_NAMES.index("category")] == "c"
    assert FEATURE_TYPES.count("c") == 1
    assert len(CATEGORIES) == 14
    assert "gender" not in FEATURE_NAMES
    assert "age" not in FEATURE_NAMES


def test_category_codes() -> None:
    assert category_code("entertainment") == 0
    assert category_code("travel") == 13
    assert category_code("crypto") == UNKNOWN_CATEGORY == 14


class TestFirstTransaction:
    def test_history_features_are_missing(self) -> None:
        f = named(FeatureEngine().process(tx()))
        for name in (
            "km_from_prev",
            "speed_kmh_from_prev",
            "secs_since_prev",
            "amount_zscore",
            "amount_to_card_mean",
            "amount_to_7d_mean",
            "hour_deviation",
        ):
            assert math.isnan(f[name]), name
        for name in (
            "tx_count_1h",
            "tx_count_24h",
            "amount_sum_24h",
            "amount_sum_1h",
            "tx_count_7d",
            "amount_sum_7d",
            "max_amount_24h",
            "distinct_categories_24h",
        ):
            assert f[name] == 0, name
        assert f["prior_tx_count"] == 0
        assert f["category_first_use"] == 1

    def test_static_features(self) -> None:
        f = named(FeatureEngine().process(tx(amount=99.0, category="travel")))
        assert f["amount"] == 99.0
        assert f["log_amount"] == pytest.approx(math.log1p(99.0))
        assert f["category"] == 13
        assert f["hour"] == 12
        assert f["day_of_week"] == 0  # Monday

    def test_without_profile_home_features_are_missing(self) -> None:
        f = named(FeatureEngine().process(tx()))
        assert math.isnan(f["km_from_home"])
        assert math.isnan(f["log_city_pop"])

    def test_with_profile(self) -> None:
        profile = CardProfile(
            card_id=CARD, home_lat=TORONTO[0], home_long=TORONTO[1], city_pop=2_700_000
        )
        f = named(FeatureEngine({CARD: profile}).process(tx(where=MONTREAL)))
        assert f["km_from_home"] == pytest.approx(504, abs=5)
        assert f["log_city_pop"] == pytest.approx(math.log1p(2_700_000))


class TestHistory:
    def test_features_use_only_prior_transactions(self) -> None:
        eng = FeatureEngine()
        eng.process(tx(0, amount=10))
        f = named(eng.process(tx(30, amount=1000)))
        assert f["tx_count_1h"] == 1
        assert f["amount_sum_24h"] == 10  # the current 1000 is not included
        assert f["prior_tx_count"] == 1
        assert f["secs_since_prev"] == 1800

    def test_window_boundaries_are_exclusive(self) -> None:
        eng = FeatureEngine()
        eng.process(tx(0))
        assert named(eng.process(tx(60)))["tx_count_1h"] == 0  # exactly 1 h ago: out
        eng2 = FeatureEngine()
        eng2.process(tx(0))
        assert named(eng2.process(tx(59.99)))["tx_count_1h"] == 1
        eng3 = FeatureEngine()
        eng3.process(tx(0, amount=7))
        f = named(eng3.process(tx(24 * 60)))  # exactly 24 h ago: out
        assert f["tx_count_24h"] == 0
        assert f["amount_sum_24h"] == 0

    def test_velocity_burst(self) -> None:
        eng = FeatureEngine()
        for i in range(5):
            eng.process(tx(i * 5, amount=100, n=i))
        f = named(eng.process(tx(25, n=9)))
        assert f["tx_count_1h"] == 5
        assert f["amount_sum_24h"] == 500

    def test_distance_and_speed_from_previous(self) -> None:
        eng = FeatureEngine()
        eng.process(tx(0, where=TORONTO))
        f = named(eng.process(tx(60, where=MONTREAL)))
        km = haversine_km(*TORONTO, *MONTREAL)
        assert f["km_from_prev"] == pytest.approx(km)
        assert f["speed_kmh_from_prev"] == pytest.approx(km)  # 1 hour apart

    def test_speed_capped_for_simultaneous_far_purchases(self) -> None:
        eng = FeatureEngine()
        eng.process(tx(0, where=TORONTO))
        f = named(eng.process(tx(0, where=(-33.86, 151.2))))  # Sydney, same second
        assert f["speed_kmh_from_prev"] == MAX_SPEED_KMH
        assert f["secs_since_prev"] == 0

    def test_zscore_against_card_history(self) -> None:
        eng = FeatureEngine()
        for i, a in enumerate([10.0, 20.0, 30.0]):
            eng.process(tx(i * 600, amount=a, n=i))
        f = named(eng.process(tx(3000, amount=40.0, n=5)))
        assert f["amount_zscore"] == pytest.approx((40 - 20) / 10)

    def test_zscore_zero_variance(self) -> None:
        eng = FeatureEngine()
        eng.process(tx(0, amount=5, n=1))
        eng.process(tx(1, amount=5, n=2))
        assert named(eng.process(tx(2, amount=500, n=3)))["amount_zscore"] == 0.0

    def test_seven_day_and_spree_features(self) -> None:
        eng = FeatureEngine()
        eng.process(tx(0, amount=100, category="grocery_pos", n=1))  # 3 days before
        eng.process(tx(3 * 24 * 60 - 120, amount=40, category="travel", n=2))  # 2 h before
        eng.process(tx(3 * 24 * 60 - 30, amount=60, category="misc_net", n=3))  # 30 min before
        f = named(eng.process(tx(3 * 24 * 60, amount=300, category="misc_net", n=4)))
        assert f["tx_count_7d"] == 3
        assert f["amount_sum_7d"] == 200
        assert f["tx_count_24h"] == 2
        assert f["max_amount_24h"] == 60
        assert f["distinct_categories_24h"] == 2
        assert f["amount_sum_1h"] == 60
        assert f["amount_to_7d_mean"] == pytest.approx(300 / (200 / 3))
        assert f["amount_to_card_mean"] == pytest.approx(300 / (200 / 3))

    def test_seven_day_boundary_is_exclusive(self) -> None:
        eng = FeatureEngine()
        eng.process(tx(0, amount=5))
        assert named(eng.process(tx(7 * 24 * 60)))["tx_count_7d"] == 0

    @pytest.mark.parametrize(("usual_hour", "now_hour", "expected"), [
        (12, 12, 0.0), (12, 15, 3.0), (23, 1, 2.0), (1, 23, 2.0), (0, 12, 12.0),
    ])  # fmt: skip
    def test_hour_deviation_wraps_midnight(
        self, usual_hour: int, now_hour: int, expected: float
    ) -> None:
        eng = FeatureEngine()
        day = 24 * 60
        eng.process(tx(usual_hour * 60 - 12 * 60, n=1))  # T0 is 12:00
        f = named(eng.process(tx(2 * day + now_hour * 60 - 12 * 60, n=2)))
        assert f["hour_deviation"] == pytest.approx(expected, abs=1e-9)

    def test_category_first_use(self) -> None:
        eng = FeatureEngine()
        eng.process(tx(0, category="grocery_pos"))
        assert named(eng.process(tx(1, category="grocery_pos")))["category_first_use"] == 0
        assert named(eng.process(tx(2, category="travel")))["category_first_use"] == 1

    def test_cards_are_isolated(self) -> None:
        eng = FeatureEngine()
        eng.process(tx(0, card=CARD))
        f = named(eng.process(tx(1, card=OTHER)))
        assert f["prior_tx_count"] == 0
        assert eng.card_count == 2

    def test_late_event_is_clamped_to_latest_time(self) -> None:
        eng = FeatureEngine()
        eng.process(tx(120))
        f = named(eng.process(tx(60)))  # arrives late, 1 h before the last one
        assert f["secs_since_prev"] == 0
        state = eng.state(CARD)
        assert state is not None
        assert state.last_ts == (T0 + timedelta(minutes=120)).timestamp()

    def test_old_window_entries_are_evicted(self) -> None:
        eng = FeatureEngine()
        for i in range(10):
            eng.process(tx(i, n=i))
        eng.process(tx(8 * 24 * 60, n=99))
        state = eng.state(CARD)
        assert state is not None
        assert len(state.window) == 1


@pytest.mark.parametrize(
    ("ts", "amount"), [(math.nan, 1.0), (math.inf, 1.0), (0.0, 0.0), (0.0, -5.0), (0.0, math.nan)]
)
def test_rejects_invalid_raw_input(ts: float, amount: float) -> None:
    with pytest.raises(ValueError, match="invalid transaction"):
        FeatureEngine().process_raw(
            card_id=CARD, ts=ts, amount=amount, category=0, merch_lat=0, merch_long=0
        )


def test_process_and_process_raw_agree() -> None:
    a, b = FeatureEngine(), FeatureEngine()
    for i, minutes in enumerate([0, 3, 50, 400]):
        event = tx(minutes, amount=10 + i, category=CATEGORIES[i], n=i)
        via_model = a.process(event)
        via_raw = b.process_raw(
            card_id=event.card_id,
            ts=event.event_time.timestamp(),
            amount=event.amount,
            category=category_code(event.category),
            merch_lat=event.merch_lat,
            merch_long=event.merch_long,
        )
        assert via_model == pytest.approx(via_raw, nan_ok=True)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.floats(0, 10 * 86400),  # seconds offset, any order
            st.floats(0.01, 50_000),
            st.integers(0, UNKNOWN_CATEGORY),
            st.floats(-90, 90),
            st.floats(-180, 180),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_invariants_hold_for_any_sequence(
    events: list[tuple[float, float, int, float, float]],
) -> None:
    eng = FeatureEngine()
    for i, (offset, amount, cat, lat, lon) in enumerate(events):
        f = named(
            eng.process_raw(
                card_id=CARD,
                ts=1.6e9 + offset,
                amount=amount,
                category=cat,
                merch_lat=lat,
                merch_long=lon,
            )
        )
        assert f["prior_tx_count"] == i
        assert 0 <= f["tx_count_1h"] <= f["tx_count_24h"] <= f["tx_count_7d"] <= i
        assert 0 <= f["amount_sum_1h"] <= f["amount_sum_24h"] * (1 + 1e-9) + 1e-9
        assert f["amount_sum_24h"] <= f["amount_sum_7d"] * (1 + 1e-9) + 1e-9
        assert 0 <= f["distinct_categories_24h"] <= f["tx_count_24h"]
        assert f["max_amount_24h"] <= f["amount_sum_24h"] + 1e-6
        if i:
            assert 0 <= f["hour_deviation"] <= 12 + 1e-9
            assert f["amount_to_card_mean"] > 0
        assert f["amount_sum_24h"] >= 0
        assert 0 <= f["hour"] <= 23
        assert 0 <= f["day_of_week"] <= 6
        if i:
            assert f["secs_since_prev"] >= 0
            assert 0 <= f["speed_kmh_from_prev"] <= MAX_SPEED_KMH
            assert 0 <= f["km_from_prev"] <= 20_016  # half Earth's circumference
        for name, value in f.items():
            assert not math.isinf(value), name
