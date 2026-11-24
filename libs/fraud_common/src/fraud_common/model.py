"""Loading and running the trained fraud model.

The artifact directory holds `model.ubj` (XGBoost binary) and `metadata.json`. Loading
refuses to start when the file hash, the feature list or the category vocabulary differ
from what this code computes: a model trained on different features must never score
live traffic.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import xgboost as xgb
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from fraud_common.features import CATEGORIES, FEATURE_NAMES, FEATURE_TYPES, N_FEATURES

MODEL_FILE: Final = "model.ubj"
METADATA_FILE: Final = "metadata.json"

FloatMatrix = npt.NDArray[np.float32]
FloatVector = npt.NDArray[np.float64]


class ModelError(RuntimeError):
    """The model artifact is missing, corrupt or incompatible with this code."""


class SplitMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: int = Field(ge=0)
    positives: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    pr_auc: float = Field(ge=0, le=1)
    roc_auc: float = Field(ge=0, le=1)
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    tn: int = Field(ge=0)
    precision_ci95: tuple[float, float] | None = None
    recall_ci95: tuple[float, float] | None = None


class ModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    model_version: str = Field(pattern=r"^fraud_xgb-[0-9a-f]{12}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_names: tuple[str, ...]
    feature_types: tuple[str, ...]
    category_vocab: tuple[str, ...]
    threshold: float = Field(gt=0, lt=1)
    threshold_rule: str = Field(min_length=1)
    target_precision: float = Field(gt=0, le=1)
    best_iteration: int = Field(ge=0)
    trained_at: AwareDatetime
    xgboost_version: str
    data_source: str
    splits: dict[str, str]
    metrics: dict[str, SplitMetrics]
    params: dict[str, str | int | float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    """Write via temp file + rename so a crash never leaves a half-written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).chmod(0o644)  # mkstemp makes 0600; services run as another user
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def feature_matrix(rows: list[list[float]]) -> FloatMatrix:
    """Stack feature vectors into the float32 matrix the model expects."""
    if not rows:
        return np.empty((0, N_FEATURES), dtype=np.float32)
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != N_FEATURES:
        raise ValueError(f"expected rows of {N_FEATURES} features, got shape {matrix.shape}")
    return matrix


def make_dmatrix(x: FloatMatrix, y: npt.NDArray[np.int8] | None = None) -> xgb.DMatrix:
    return xgb.DMatrix(
        x,
        label=y,
        feature_names=list(FEATURE_NAMES),
        feature_types=list(FEATURE_TYPES),
        enable_categorical=True,
        missing=math.nan,
    )


class FraudModel:
    def __init__(self, booster: xgb.Booster, metadata: ModelMetadata) -> None:
        self._booster = booster
        self.metadata = metadata

    @property
    def version(self) -> str:
        return self.metadata.model_version

    @property
    def threshold(self) -> float:
        return self.metadata.threshold

    @classmethod
    def load(cls, model_dir: Path) -> FraudModel:
        model_path, meta_path = model_dir / MODEL_FILE, model_dir / METADATA_FILE
        for p in (model_path, meta_path):
            if not p.is_file():
                raise ModelError(f"missing model artifact file: {p}")
        try:
            metadata = ModelMetadata.model_validate_json(meta_path.read_bytes())
        except ValueError as exc:
            raise ModelError(f"invalid {METADATA_FILE}: {exc}") from exc

        actual_sha = sha256_file(model_path)
        if actual_sha != metadata.artifact_sha256:
            raise ModelError(
                f"{MODEL_FILE} sha256 {actual_sha[:12]}... does not match metadata "
                f"{metadata.artifact_sha256[:12]}...; the file was changed or corrupted"
            )
        problems = []
        if metadata.feature_names != FEATURE_NAMES:
            problems.append("feature names differ from fraud_common.features")
        if metadata.feature_types != FEATURE_TYPES:
            problems.append("feature types differ from fraud_common.features")
        if metadata.category_vocab != CATEGORIES:
            problems.append("category vocabulary differs from fraud_common.features")
        if problems:
            raise ModelError("incompatible model: " + "; ".join(problems))

        booster = xgb.Booster()
        try:
            booster.load_model(str(model_path))
        except xgb.core.XGBoostError as exc:
            raise ModelError(f"could not load {MODEL_FILE}: {exc}") from exc
        if tuple(booster.feature_names or ()) != FEATURE_NAMES:
            raise ModelError("booster feature names differ from metadata")
        return cls(booster, metadata)

    def predict(self, x: FloatMatrix) -> FloatVector:
        """Fraud probability per row, in input order."""
        if x.ndim != 2 or x.shape[1] != N_FEATURES:
            raise ValueError(f"expected shape (n, {N_FEATURES}), got {x.shape}")
        if x.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        scores = self._booster.inplace_predict(
            x,
            iteration_range=(0, self.metadata.best_iteration + 1),
            validate_features=False,
        )
        return np.asarray(scores, dtype=np.float64)

    def set_threads(self, threads: int) -> None:
        """Cap XGBoost's threads. Inside a container, XGBoost sees every host CPU; more
        threads than the CPU quota get throttled and add long tail latency."""
        if threads < 1:
            raise ValueError("threads must be >= 1")
        self._booster.set_param({"nthread": threads})

    def flag(self, scores: FloatVector) -> npt.NDArray[np.bool_]:
        return scores >= self.metadata.threshold
