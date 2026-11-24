"""Ingest the Kaggle Sparkov dataset (fraudTrain.csv + fraudTest.csv).

Input: the downloaded zip, or a folder containing both CSVs.
Output (in `out_dir`):
  transactions.parquet  one row per transaction, time-sorted, no personal data
  profiles.parquet      one row per card: home location and city population

Privacy: names, street, city, zip, job, gender and date of birth are dropped here and
never reach the rest of the system. The card number becomes a SHA-256 token.
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from fraud_common.schemas import MAX_AMOUNT

log = logging.getLogger(__name__)

SOURCE_FILES: Final = {"train": "fraudTrain.csv", "test": "fraudTest.csv"}

RAW_DTYPES: Final[dict[str, str]] = {
    "trans_date_trans_time": "string",
    "cc_num": "string",
    "merchant": "string",
    "category": "string",
    "amt": "float64",
    "lat": "float64",
    "long": "float64",
    "city_pop": "int64",
    "trans_num": "string",
    "merch_lat": "float64",
    "merch_long": "float64",
    "is_fraud": "int8",
}
TRANSACTION_COLUMNS: Final = (
    "tx_id",
    "card_id",
    "merchant",
    "category",
    "amount",
    "event_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
    "split_source",
)
PROFILE_COLUMNS: Final = ("card_id", "home_lat", "home_long", "city_pop")


class IngestError(ValueError):
    """The input does not look like the Sparkov dataset."""


@dataclass(frozen=True, slots=True)
class IngestReport:
    rows_in: int
    rows_out: int
    dropped_invalid: int
    dropped_duplicate: int
    cards: int
    profile_conflicts: int
    fraud_rate: float


def card_token(cc_num: str) -> str:
    """Stable 64-bit token for a card number (16 hex chars)."""
    return hashlib.sha256(cc_num.strip().encode()).hexdigest()[:16]


def _open_sources(source: Path) -> dict[str, bytes]:
    if source.is_dir():
        files = {k: source / v for k, v in SOURCE_FILES.items()}
        missing = [str(p) for p in files.values() if not p.is_file()]
        if missing:
            raise IngestError(f"missing in {source}: {', '.join(missing)}")
        return {k: p.read_bytes() for k, p in files.items()}
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            by_name = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
            out = {}
            for key, fname in SOURCE_FILES.items():
                if fname not in by_name:
                    raise IngestError(f"{fname} not found inside {source.name}")
                out[key] = zf.read(by_name[fname])
            return out
    raise IngestError(f"{source} is neither a folder nor a zip file")


def read_raw(source: Path) -> pd.DataFrame:
    frames = []
    for split_source, blob in _open_sources(source).items():
        header = pd.read_csv(io.BytesIO(blob), nrows=0).columns
        missing = sorted(set(RAW_DTYPES) - set(header))
        if missing:
            raise IngestError(f"{SOURCE_FILES[split_source]} lacks columns: {missing}")
        df = pd.read_csv(
            io.BytesIO(blob), usecols=list(RAW_DTYPES), dtype=RAW_DTYPES, keep_default_na=True
        )
        df["split_source"] = split_source
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, IngestReport]:
    rows_in = len(raw)
    df = raw.copy()
    for col in ("amt", "lat", "long", "merch_lat", "merch_long", "city_pop", "is_fraud"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["event_time"] = pd.to_datetime(
        df["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce", utc=True
    )
    df["tx_id"] = df["trans_num"].str.strip().str.lower()
    df["merchant"] = df["merchant"].str.strip().str.removeprefix("fraud_")
    df["category"] = df["category"].str.strip().str.lower()

    valid = (
        df["event_time"].notna()
        & df["tx_id"].str.fullmatch(r"[0-9a-f]{32}").fillna(False).astype(bool)
        & df["cc_num"].notna()
        & df["merchant"].str.len().between(1, 128).fillna(False).astype(bool)
        & df["category"].str.fullmatch(r"[a-z][a-z_]{0,63}").fillna(False).astype(bool)
        & np.isfinite(df["amt"])
        & (df["amt"] > 0)
        & (df["amt"] <= MAX_AMOUNT)
        & df["is_fraud"].isin([0, 1])
        & df["city_pop"].notna()
        & (df["city_pop"] >= 0)
    )
    for col, bound in (("lat", 90), ("merch_lat", 90), ("long", 180), ("merch_long", 180)):
        valid &= df[col].between(-bound, bound)
    dropped_invalid = int((~valid).sum())
    df = df[valid].astype({"is_fraud": "int8", "city_pop": "int64"})

    before = len(df)
    df = df.drop_duplicates(subset="tx_id", keep="first")
    dropped_duplicate = before - len(df)

    df["card_id"] = df["cc_num"].map(card_token)
    if df.groupby("card_id")["cc_num"].nunique().max() > 1:
        raise IngestError("card token collision; widen the token")

    # One profile per card; Sparkov keeps them constant, so conflicts mean bad data.
    prof = df[["card_id", "lat", "long", "city_pop", "event_time"]].sort_values("event_time")
    conflicts = int(
        (prof.groupby("card_id")[["lat", "long", "city_pop"]].nunique() > 1).any(axis=1).sum()
    )
    profiles = (
        prof.groupby("card_id", as_index=False)
        .last()
        .rename(columns={"lat": "home_lat", "long": "home_long"})[list(PROFILE_COLUMNS)]
    )

    df = df.rename(columns={"amt": "amount"})
    df = df.sort_values(["event_time", "tx_id"], kind="mergesort").reset_index(drop=True)
    transactions = df[list(TRANSACTION_COLUMNS)]
    report = IngestReport(
        rows_in=rows_in,
        rows_out=len(transactions),
        dropped_invalid=dropped_invalid,
        dropped_duplicate=dropped_duplicate,
        cards=len(profiles),
        profile_conflicts=conflicts,
        fraud_rate=float(transactions["is_fraud"].mean()) if len(transactions) else 0.0,
    )
    if report.rows_out == 0:
        raise IngestError("no valid rows left after cleaning")
    return transactions, profiles, report


def ingest(source: Path, out_dir: Path) -> IngestReport:
    transactions, profiles, report = clean(read_raw(source))
    out_dir.mkdir(parents=True, exist_ok=True)
    transactions.to_parquet(out_dir / "transactions.parquet", index=False)
    profiles.to_parquet(out_dir / "profiles.parquet", index=False)
    log.info("ingest finished", extra={"report": report})
    return report
