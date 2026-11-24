import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_common.features import FeatureEngine
from fraud_training.dataset import build_features, split
from fraud_training.ingest import IngestError, card_token, clean, ingest, read_raw
from fraud_training.train import choose_threshold, evaluate
from tests.fixtures.sparkov_like import KAGGLE_COLUMNS, write_kaggle_zip


@pytest.fixture(scope="module")
def kaggle_zip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_kaggle_zip(tmp_path_factory.mktemp("kaggle") / "archive.zip", n_cards=15)


@pytest.fixture(scope="module")
def cleaned(kaggle_zip: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx, profiles, _ = clean(read_raw(kaggle_zip))
    return tx, profiles


# ------------------------------------------------------------------ ingest


def test_card_token() -> None:
    assert card_token("4000123412341234") == card_token(" 4000123412341234 ")
    assert len(card_token("1")) == 16
    assert card_token("1") != card_token("2")


def test_ingest_drops_personal_data(cleaned: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    tx, profiles = cleaned
    pii = {"first", "last", "street", "city", "zip", "job", "dob", "gender", "cc_num"}
    assert not pii & set(tx.columns)
    assert not pii & set(profiles.columns)
    assert tx["card_id"].str.fullmatch(r"[0-9a-f]{16}").all()


def test_ingest_output_shape(cleaned: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    tx, profiles = cleaned
    assert tx["event_time"].is_monotonic_increasing
    assert str(tx["event_time"].dt.tz) == "UTC"
    assert tx["tx_id"].is_unique
    assert not tx["merchant"].str.startswith("fraud_").any()
    assert set(tx["split_source"]) == {"train", "test"}
    assert profiles["card_id"].is_unique
    assert set(tx["card_id"]) == set(profiles["card_id"])


def test_ingest_writes_parquet(kaggle_zip: Path, tmp_path: Path) -> None:
    report = ingest(kaggle_zip, tmp_path)
    assert report.rows_out == len(pd.read_parquet(tmp_path / "transactions.parquet"))
    assert report.cards == 15
    assert report.dropped_invalid == report.dropped_duplicate == report.profile_conflicts == 0
    assert 0 < report.fraud_rate < 0.05


def test_ingest_accepts_folder(kaggle_zip: Path, tmp_path: Path) -> None:
    with zipfile.ZipFile(kaggle_zip) as zf:
        zf.extractall(tmp_path)
    assert len(read_raw(tmp_path)) == len(read_raw(kaggle_zip))


def _raw_rows(kaggle_zip: Path) -> pd.DataFrame:
    return read_raw(kaggle_zip).head(50).copy()


@pytest.mark.parametrize(
    ("column", "bad"),
    [
        ("amt", -3.0),
        ("amt", 0.0),
        ("amt", float("inf")),
        ("amt", "abc"),
        ("is_fraud", 2),
        ("city_pop", -1),
        ("trans_date_trans_time", "yesterday"),
        ("trans_num", "xyz"),
        ("merch_lat", 95.0),
        ("long", -200.0),
        ("category", "Bad Cat"),
    ],
)
def test_invalid_rows_are_dropped(kaggle_zip: Path, column: str, bad: object) -> None:
    raw = _raw_rows(kaggle_zip)
    values = raw[column].astype(object).tolist()
    values[0] = bad
    raw[column] = values
    _, _, report = clean(raw)
    assert report.dropped_invalid == 1
    assert report.rows_out == 49


def test_duplicates_are_dropped(kaggle_zip: Path) -> None:
    raw = _raw_rows(kaggle_zip)
    _, _, report = clean(pd.concat([raw, raw.head(5)], ignore_index=True))
    assert report.dropped_duplicate == 5


def test_profile_conflict_is_counted(kaggle_zip: Path) -> None:
    raw = _raw_rows(kaggle_zip)
    pops = raw["city_pop"].tolist()
    pops[0] += 1
    raw["city_pop"] = pops
    assert clean(raw)[2].profile_conflicts == 1


def test_all_invalid_is_an_error(kaggle_zip: Path) -> None:
    raw = _raw_rows(kaggle_zip)
    raw["amt"] = -1.0
    with pytest.raises(IngestError, match="no valid rows"):
        clean(raw)


def test_missing_column_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as zf:
        cols = [c for c in KAGGLE_COLUMNS if c != "amt"]
        for name in ("fraudTrain.csv", "fraudTest.csv"):
            zf.writestr(name, ",".join(cols) + "\n")
    with pytest.raises(IngestError, match=r"lacks columns"):
        read_raw(path)


def test_missing_file_in_zip(tmp_path: Path) -> None:
    path = tmp_path / "half.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("fraudTrain.csv", "a\n")
    with pytest.raises(IngestError, match=r"fraudTest\.csv not found"):
        read_raw(path)


def test_zip_with_nested_folder(kaggle_zip: Path, tmp_path: Path) -> None:
    nested = tmp_path / "nested.zip"
    with zipfile.ZipFile(kaggle_zip) as src, zipfile.ZipFile(nested, "w") as dst:
        for name in src.namelist():
            dst.writestr(f"archive/{name}", src.read(name))
    assert len(read_raw(nested)) == len(read_raw(kaggle_zip))


@pytest.mark.parametrize("kind", ["missing_dir", "not_zip", "empty_dir"])
def test_bad_source(tmp_path: Path, kind: str) -> None:
    if kind == "missing_dir":
        source = tmp_path / "nope"
    elif kind == "not_zip":
        source = tmp_path / "x.zip"
        source.write_bytes(b"plain text")
    else:
        source = tmp_path
    with pytest.raises(IngestError):
        read_raw(source)


# ------------------------------------------------------------------ dataset


def test_build_features_requires_time_order(cleaned: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    tx, _ = cleaned
    with pytest.raises(ValueError, match="sorted"):
        build_features(tx.iloc[::-1], FeatureEngine())


def test_split_is_time_ordered_and_complete(cleaned: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    tx, _ = cleaned
    x = build_features(tx, FeatureEngine())
    parts = split(tx, x)
    assert sum(p.rows for p in parts.values()) == len(tx)
    assert parts["train"].end < parts["val"].start
    assert parts["val"].end <= parts["test"].start
    assert parts["val"].rows == pytest.approx(
        0.15 * (parts["train"].rows + parts["val"].rows), rel=0.05
    )


def test_split_validation(cleaned: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    tx, _ = cleaned
    x = build_features(tx, FeatureEngine())
    with pytest.raises(ValueError, match="val_fraction"):
        split(tx, x, val_fraction=1.0)
    with pytest.raises(ValueError, match="length"):
        split(tx, x[:-1])
    only_train = tx[tx["split_source"] == "train"]
    with pytest.raises(ValueError, match="both"):
        split(only_train, x[: len(only_train)])


# ------------------------------------------------------------------ threshold + metrics


def test_choose_threshold_meets_target() -> None:
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1], dtype=np.int8)
    s = np.linspace(0.05, 0.95, 10)
    t = choose_threshold(y, s, target_precision=0.8)
    flagged = s >= t.value
    assert y[flagged].mean() >= 0.8
    assert t.value == pytest.approx(s[4])  # lowest cut that keeps precision >= 0.8
    assert "max recall" in t.rule


def test_choose_threshold_falls_back_to_f1() -> None:
    y = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    t = choose_threshold(y, s, target_precision=0.99)
    assert "unreachable" in t.rule


@pytest.mark.parametrize("y", [[0, 0, 0], [1, 1, 1]])
def test_choose_threshold_needs_both_classes(y: list[int]) -> None:
    with pytest.raises(ValueError, match="both classes"):
        choose_threshold(np.array(y, dtype=np.int8), np.array([0.1, 0.5, 0.9]), 0.9)


def test_choose_threshold_rejects_bad_target() -> None:
    with pytest.raises(ValueError, match="target_precision"):
        choose_threshold(np.array([0, 1], np.int8), np.array([0.1, 0.9]), 0.0)


def test_evaluate_counts() -> None:
    y = np.array([1, 1, 0, 0, 1], dtype=np.int8)
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.7])
    m = evaluate(y, s, 0.5)
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 1, 1)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.precision_ci95 is None


def test_evaluate_nothing_flagged() -> None:
    m = evaluate(np.array([0, 1], np.int8), np.array([0.1, 0.2]), 0.9, with_ci=True)
    assert m.precision == m.recall == m.f1 == 0
    assert m.precision_ci95 == (0.0, 0.0)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(np.int8)
    s = np.clip(y * 0.6 + rng.random(5000) * 0.5, 0, 1)
    m = evaluate(y, s, 0.6, with_ci=True)
    assert m.precision_ci95 is not None
    assert m.recall_ci95 is not None
    assert m.precision_ci95[0] <= m.precision <= m.precision_ci95[1]
    assert m.recall_ci95[0] <= m.recall <= m.recall_ci95[1]
    again = evaluate(y, s, 0.6, with_ci=True)
    assert again.precision_ci95 == m.precision_ci95  # seeded, reproducible


def test_read_raw_handles_bytes_io_roundtrip() -> None:
    # guards against pandas treating "NA"-like merchant names as missing
    df = pd.read_csv(io.StringIO("merchant\nNA Shop\n"), dtype={"merchant": "string"})
    assert df["merchant"].iloc[0] == "NA Shop"
