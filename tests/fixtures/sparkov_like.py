"""Small synthetic dataset in the exact Kaggle Sparkov CSV format, for tests only.

Normal behaviour: a few daytime purchases per day near home. Fraud: occasional night-time
sprees of large online purchases far from home, in quick succession. That mirrors the
patterns in the real dataset closely enough to test the pipeline end to end, but the
metrics on it mean nothing; real numbers come only from the Kaggle data.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_common.features import CATEGORIES

KAGGLE_COLUMNS = [
    "Unnamed: 0", "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first",
    "last", "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob",
    "trans_num", "unix_time", "merch_lat", "merch_long", "is_fraud",
]  # fmt: skip

FRAUD_CATEGORIES = ("shopping_net", "misc_net", "grocery_pos")


def generate(
    *, n_cards: int = 40, start: datetime, days: int, seed: int, fraud_spree_prob: float = 0.004
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for card in range(n_cards):
        cc_num = str(4_000_000_000_000 + seed * 10_000 + card)
        home_lat, home_long = rng.uniform(30, 47), rng.uniform(-120, -75)
        city_pop = int(rng.integers(500, 2_000_000))
        for day in range(days):
            date = start + timedelta(days=day)
            for _ in range(int(rng.poisson(2.5))):
                ts = date + timedelta(seconds=int(rng.integers(7 * 3600, 22 * 3600)))
                cat = CATEGORIES[int(rng.integers(len(CATEGORIES)))]
                rows.append(_row(rng, cc_num, ts, cat, float(np.exp(rng.normal(3.5, 0.9))),
                                 home_lat, home_long, city_pop, 0, spread=0.8))  # fmt: skip
            if rng.random() < fraud_spree_prob:
                t = date + timedelta(hours=float(rng.uniform(22, 27)))
                far_lat, far_long = home_lat + rng.uniform(5, 10), home_long + rng.uniform(5, 15)
                for _ in range(int(rng.integers(3, 8))):
                    t += timedelta(minutes=float(rng.uniform(2, 25)))
                    cat = FRAUD_CATEGORIES[int(rng.integers(len(FRAUD_CATEGORIES)))]
                    rows.append(_row(rng, cc_num, t, cat, float(rng.uniform(250, 1200)),
                                     far_lat, far_long, city_pop, 1, spread=0.3,
                                     home=(home_lat, home_long)))  # fmt: skip
    df = pd.DataFrame(rows).sort_values("trans_date_trans_time", kind="mergesort")
    df.insert(0, "Unnamed: 0", range(len(df)))
    return df[KAGGLE_COLUMNS].reset_index(drop=True)


def _row(
    rng: np.random.Generator,
    cc_num: str,
    ts: datetime,
    category: str,
    amount: float,
    lat: float,
    lon: float,
    city_pop: int,
    is_fraud: int,
    *,
    spread: float,
    home: tuple[float, float] | None = None,
) -> dict[str, object]:
    home_lat, home_long = home or (lat, lon)
    tx = hashlib.md5(f"{cc_num}{ts.isoformat()}{rng.random()}".encode(), usedforsecurity=False)
    return {
        "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": cc_num,
        "merchant": f"fraud_Merchant {int(rng.integers(200))}",
        "category": category,
        "amt": round(amount, 2),
        "first": "Jane",
        "last": "Doe",
        "gender": "F",
        "street": "1 Main St",
        "city": "Town",
        "state": "ON",
        "zip": 12345,
        "lat": round(home_lat, 4),
        "long": round(home_long, 4),
        "city_pop": city_pop,
        "job": "Engineer",
        "dob": "1990-01-01",
        "trans_num": tx.hexdigest(),
        "unix_time": int(ts.timestamp()),
        "merch_lat": round(lat + rng.uniform(-spread, spread), 6),
        "merch_long": round(lon + rng.uniform(-spread, spread), 6),
        "is_fraud": is_fraud,
    }


def write_kaggle_zip(path: Path, *, n_cards: int = 40, seed: int = 7) -> Path:
    """Zip laid out like the Kaggle download: fraudTrain.csv + fraudTest.csv."""
    full = generate(n_cards=n_cards, start=datetime(2019, 1, 1), days=270, seed=seed)
    is_train = full["trans_date_trans_time"] < "2019-07-01"
    train, test = full[is_train], full[~is_train]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in (("fraudTrain.csv", train), ("fraudTest.csv", test)):
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            zf.writestr(name, buf.getvalue())
    return path
