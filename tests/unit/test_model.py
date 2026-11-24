import json
import math
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from fraud_common.features import CATEGORIES, FEATURE_NAMES, N_FEATURES
from fraud_common.model import (
    METADATA_FILE,
    MODEL_FILE,
    FraudModel,
    ModelError,
    feature_matrix,
    make_dmatrix,
    sha256_file,
    write_atomic,
)
from tests.fixtures.model_artifact import write_artifact


def _train_tiny(seed: int = 0) -> xgb.Booster:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(400, N_FEATURES)).astype(np.float32)
    x[:, FEATURE_NAMES.index("category")] = rng.integers(0, 15, 400)
    y = (x[:, 0] > 0.8).astype(np.int8)
    return xgb.train(
        {"objective": "binary:logistic", "max_depth": 3, "seed": 0},
        make_dmatrix(x, y),
        num_boost_round=10,
    )


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    return write_artifact(tmp_path)


def _edit_metadata(path: Path, **changes: object) -> None:
    data = json.loads((path / METADATA_FILE).read_text())
    data.update(changes)
    (path / METADATA_FILE).write_text(json.dumps(data))


def test_load_and_predict(artifact: Path) -> None:
    model = FraudModel.load(artifact)
    x = np.zeros((3, N_FEATURES), dtype=np.float32)
    x[0, 0] = 5.0
    scores = model.predict(x)
    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores[0] > scores[1]
    assert model.flag(scores).dtype == np.bool_
    assert model.version.startswith("fraud_xgb-")
    assert model.threshold == 0.5


def test_inplace_predict_matches_dmatrix_predict(artifact: Path) -> None:
    model = FraudModel.load(artifact)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, N_FEATURES)).astype(np.float32)
    x[:, FEATURE_NAMES.index("category")] = rng.integers(0, 15, 50)
    x[::5, 7] = math.nan  # missing values take the learned default branch
    booster = xgb.Booster()
    booster.load_model(str(artifact / MODEL_FILE))
    expected = booster.predict(make_dmatrix(x), iteration_range=(0, 10))
    np.testing.assert_allclose(model.predict(x), expected, rtol=1e-6)


def test_predict_empty_batch(artifact: Path) -> None:
    assert FraudModel.load(artifact).predict(np.empty((0, N_FEATURES), np.float32)).shape == (0,)


@pytest.mark.parametrize("shape", [(2, N_FEATURES - 1), (2, N_FEATURES + 1), (N_FEATURES,)])
def test_predict_rejects_wrong_shape(artifact: Path, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="expected shape"):
        FraudModel.load(artifact).predict(np.zeros(shape, np.float32))


def test_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="missing"):
        FraudModel.load(tmp_path)


def test_tampered_model_is_rejected(artifact: Path) -> None:
    with (artifact / MODEL_FILE).open("ab") as fh:
        fh.write(b"\x00")
    with pytest.raises(ModelError, match="sha256"):
        FraudModel.load(artifact)


def test_corrupt_metadata_is_rejected(artifact: Path) -> None:
    (artifact / METADATA_FILE).write_text("{not json")
    with pytest.raises(ModelError, match="invalid"):
        FraudModel.load(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("feature_names", list(reversed(FEATURE_NAMES)), "feature names"),
        ("feature_types", ["q"] * N_FEATURES, "feature types"),
        ("category_vocab", [*CATEGORIES, "crypto"], "category vocabulary"),
    ],
)
def test_incompatible_contract_is_rejected(
    artifact: Path, field: str, value: object, message: str
) -> None:
    _edit_metadata(artifact, **{field: value})
    with pytest.raises(ModelError, match=message):
        FraudModel.load(artifact)


def test_garbage_model_file_with_matching_hash(artifact: Path) -> None:
    (artifact / MODEL_FILE).write_bytes(b"garbage")
    _edit_metadata(artifact, artifact_sha256=sha256_file(artifact / MODEL_FILE))
    with pytest.raises(ModelError, match="could not load"):
        FraudModel.load(artifact)


def test_feature_matrix() -> None:
    assert feature_matrix([]).shape == (0, N_FEATURES)
    m = feature_matrix([[1.0] * N_FEATURES, [math.nan] * N_FEATURES])
    assert m.dtype == np.float32
    assert m.shape == (2, N_FEATURES)
    with pytest.raises(ValueError, match="expected rows"):
        feature_matrix([[1.0] * 3])


def test_write_atomic_leaves_no_temp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "f.bin"
    target.write_bytes(b"old")

    def boom(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("os.fsync", boom)
    with pytest.raises(OSError, match="disk full"):
        write_atomic(target, b"new")
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["f.bin"]


@pytest.mark.parametrize(
    ("field", "value"), [("threshold", 0.0), ("threshold", 1.0), ("model_version", "v1")]
)
def test_metadata_validation(artifact: Path, field: str, value: object) -> None:
    _edit_metadata(artifact, **{field: value})
    with pytest.raises(ModelError, match="invalid"):
        FraudModel.load(artifact)
