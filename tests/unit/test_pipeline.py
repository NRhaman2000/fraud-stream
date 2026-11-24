"""End to end on the synthetic fixture: Kaggle-format zip -> artifact -> live scoring.

The key test is training/serving parity: replaying test transactions one at a time as
Kafka-style events through a fresh engine must give exactly the features training used.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_common.features import FeatureEngine
from fraud_common.model import FraudModel, feature_matrix
from fraud_common.schemas import TransactionEvent
from fraud_services.producer.main import Replay
from fraud_services.scorer.core import ScoringCore, load_history
from fraud_training.cli import main
from fraud_training.dataset import build_features, load_profiles
from tests.fixtures.sparkov_like import write_kaggle_zip


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("pipeline")
    source = write_kaggle_zip(root / "archive.zip", n_cards=30)
    data, models = root / "data", root / "models"
    assert main(["ingest", "--source", str(source), "--out", str(data)]) == 0
    assert main(["train", "--data", str(data), "--models", str(models)]) == 0
    return data, models


def test_artifact_loads_and_reports(
    trained: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    _, models = trained
    model = FraudModel.load(models)
    meta = model.metadata
    assert set(meta.metrics) == {"train", "val", "test"}
    assert meta.metrics["test"].precision_ci95 is not None
    assert meta.metrics["test"].pr_auc > 0.8  # fixture fraud is easy to spot
    assert main(["report", "--models", str(models)]) == 0
    assert model.version in capsys.readouterr().out


def test_training_serving_parity(trained: tuple[Path, Path]) -> None:
    data, _ = trained
    tx = pd.read_parquet(data / "transactions.parquet")
    profiles = load_profiles(pd.read_parquet(data / "profiles.parquet"))
    offline = build_features(tx, FeatureEngine(profiles))

    online_engine = FeatureEngine(profiles)
    online = []
    times = tx["event_time"].tolist()
    for rec, when in zip(tx.to_dict("records"), times, strict=True):
        event = TransactionEvent(
            tx_id=rec["tx_id"],
            card_id=rec["card_id"],
            merchant=rec["merchant"],
            category=rec["category"],
            amount=rec["amount"],
            event_time=when,
            merch_lat=rec["merch_lat"],
            merch_long=rec["merch_long"],
            produced_at_ns=0,
        )
        online.append(online_engine.process(event))
    np.testing.assert_array_equal(feature_matrix(online), offline)


def test_live_scores_match_offline_evaluation(trained: tuple[Path, Path]) -> None:
    data, models = trained
    model = FraudModel.load(models)
    tx = pd.read_parquet(data / "transactions.parquet")
    x = build_features(tx, FeatureEngine(load_profiles(pd.read_parquet(data / "profiles.parquet"))))
    test_mask = (tx["split_source"] == "test").to_numpy()
    flagged = model.flag(model.predict(x[test_mask]))
    y = tx.loc[test_mask, "is_fraud"].to_numpy()
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    m = model.metadata.metrics["test"]
    assert (tp, fp) == (m.tp, m.fp)


def test_scorer_with_history_matches_offline_test_features(trained: tuple[Path, Path]) -> None:
    """The deployed path (history at startup, then the producer's replay through the
    scorer) must give the same features, and so the same flags, as offline evaluation."""
    data, models = trained
    tx = pd.read_parquet(data / "transactions.parquet")
    profiles = load_profiles(pd.read_parquet(data / "profiles.parquet"))
    test_mask = (tx["split_source"] == "test").to_numpy()
    offline = build_features(tx, FeatureEngine(profiles))[test_mask]

    engine = FeatureEngine(profiles)
    assert load_history(data, engine) == int((~test_mask).sum())
    core = ScoringCore(engine)
    online: list[list[float]] = []
    for event, _ in Replay.load(data).iterate(loop=False):
        batch = core.prepare([(event.card_id.encode(), event.model_dump_json().encode())])
        assert batch.request is not None
        online.extend(
            [np.nan if v is None else v for v in item.features] for item in batch.request.items
        )
    live = feature_matrix(online)
    np.testing.assert_allclose(live, offline, rtol=1e-6, atol=1e-6)
    model = FraudModel.load(models)
    np.testing.assert_array_equal(
        model.flag(model.predict(live)), model.flag(model.predict(offline))
    )


def test_cli_errors_are_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ingest", "--source", str(tmp_path / "missing.zip")]) == 1
    assert main(["report", "--models", str(tmp_path)]) == 1
    assert "error:" in capsys.readouterr().err
