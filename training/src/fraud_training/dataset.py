"""Feature building and the time-based train / validation / test split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from fraud_common.features import N_FEATURES, FeatureEngine, category_code
from fraud_common.schemas import CardProfile

VAL_FRACTION: Final = 0.15


@dataclass(frozen=True, slots=True)
class Split:
    name: str
    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.int8]
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def rows(self) -> int:
        return len(self.y)

    @property
    def positives(self) -> int:
        return int(self.y.sum())


def load_profiles(profiles: pd.DataFrame) -> dict[str, CardProfile]:
    records = profiles.astype({"city_pop": "int64"}).to_dict("records")
    parsed = [CardProfile.model_validate(r) for r in records]
    return {p.card_id: p for p in parsed}


def build_features(transactions: pd.DataFrame, engine: FeatureEngine) -> npt.NDArray[np.float32]:
    """Replay every transaction, in event-time order, through the live feature engine."""
    if not transactions["event_time"].is_monotonic_increasing:
        raise ValueError("transactions must be sorted by event_time")
    ts = transactions["event_time"].astype("int64").to_numpy() / 1e9  # ns -> s
    codes = [category_code(c) for c in transactions["category"]]
    out = np.empty((len(transactions), N_FEATURES), dtype=np.float32)
    process = engine.process_raw
    for i, (card, t, amt, cat, lat, lon) in enumerate(
        zip(
            transactions["card_id"],
            ts,
            transactions["amount"].to_numpy(),
            codes,
            transactions["merch_lat"].to_numpy(),
            transactions["merch_long"].to_numpy(),
            strict=True,
        )
    ):
        out[i] = process(
            card_id=card,
            ts=float(t),
            amount=float(amt),
            category=cat,
            merch_lat=float(lat),
            merch_long=float(lon),
        )
    return out


def split(
    transactions: pd.DataFrame, x: npt.NDArray[np.float32], val_fraction: float = VAL_FRACTION
) -> dict[str, Split]:
    """train = early part of fraudTrain, val = its last `val_fraction` by time, test = fraudTest.

    The cut is a timestamp, so no moment is split across train and validation, and
    everything in validation happens after everything in training."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(x) != len(transactions):
        raise ValueError("features and transactions differ in length")
    source = transactions["split_source"].to_numpy()
    times = transactions["event_time"]
    y = transactions["is_fraud"].to_numpy(dtype=np.int8)

    train_mask_all = source == "train"
    train_times = times[train_mask_all]
    if train_times.empty or not (source == "test").any():
        raise ValueError("need rows from both fraudTrain and fraudTest")
    cut = train_times.iloc[int(len(train_times) * (1 - val_fraction))]

    masks = {
        "train": train_mask_all & (times < cut).to_numpy(),
        "val": train_mask_all & (times >= cut).to_numpy(),
        "test": source == "test",
    }
    result = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"split {name} is empty")
        result[name] = Split(
            name=name, x=x[mask], y=y[mask], start=times[mask].min(), end=times[mask].max()
        )
    if result["val"].start < result["train"].end:
        raise AssertionError("validation overlaps training in time")
    return result
