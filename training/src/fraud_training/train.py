"""Fit XGBoost, choose the alert threshold on validation, evaluate once on test, save.

The test split is touched exactly once, after every choice (hyperparameters, early
stopping, threshold) has been made on train/validation. Its numbers are what the
resume may quote.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from fraud_common.features import CATEGORIES, FEATURE_NAMES, FEATURE_TYPES
from fraud_common.model import (
    METADATA_FILE,
    MODEL_FILE,
    ModelMetadata,
    SplitMetrics,
    make_dmatrix,
    sha256_file,
    write_atomic,
)
from fraud_training.dataset import Split

log = logging.getLogger(__name__)

DEFAULT_PARAMS: Final[dict[str, str | int | float]] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "lambda": 1.0,
    "max_cat_to_onehot": 16,
    "seed": 42,
}
MAX_ROUNDS: Final = 2000
EARLY_STOPPING: Final = 100
BOOTSTRAP_SAMPLES: Final = 2000


@dataclass(frozen=True, slots=True)
class Threshold:
    value: float
    rule: str


def choose_threshold(
    y: npt.NDArray[np.int8], scores: npt.NDArray[np.float64], target_precision: float
) -> Threshold:
    """Highest-recall threshold whose precision on (y, scores) reaches the target.
    Falls back to the F1-best threshold when the target is unreachable, and says so."""
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("threshold selection needs both classes")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    # precision/recall have one more entry than thresholds (the final (1, 0) point).
    precision, recall = precision[:-1], recall[:-1]
    ok = precision >= target_precision
    if ok.any():
        candidates = np.flatnonzero(ok)
        # max recall; among equal recall, the higher threshold (fewer alerts)
        best = candidates[np.lexsort((-thresholds[candidates], -recall[candidates]))[0]]
        return Threshold(
            float(thresholds[best]), f"max recall with val precision >= {target_precision}"
        )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = int(np.argmax(f1))
    return Threshold(
        float(thresholds[best]), f"F1-best (val precision {target_precision} unreachable)"
    )


def _unit(value: float) -> float:
    """sklearn can return 1.0000000000000002; clamp float noise into [0, 1]."""
    return min(1.0, max(0.0, float(value)))


def _bootstrap_rate(hits: npt.NDArray[np.int8], rng: np.random.Generator) -> tuple[float, float]:
    """95% percentile CI of the mean of a 0/1 vector."""
    if len(hits) == 0:
        return (0.0, 0.0)
    idx = rng.integers(0, len(hits), size=(BOOTSTRAP_SAMPLES, len(hits)), dtype=np.int32)
    means = hits[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def evaluate(
    y: npt.NDArray[np.int8],
    scores: npt.NDArray[np.float64],
    threshold: float,
    *,
    with_ci: bool = False,
    seed: int = 42,
) -> SplitMetrics:
    flagged = scores >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both = 0 < y.sum() < len(y)
    metrics = SplitMetrics(
        rows=len(y),
        positives=int(y.sum()),
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=_unit(average_precision_score(y, scores)) if both else 0.0,
        roc_auc=_unit(roc_auc_score(y, scores)) if both else 0.0,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )
    if not with_ci:
        return metrics
    # Precision is the fraud rate among flagged rows, recall the flag rate among frauds,
    # so each CI is a bootstrap of one 0/1 vector: exact and cheap on 555k rows.
    rng = np.random.default_rng(seed)
    return metrics.model_copy(
        update={
            "precision_ci95": _bootstrap_rate(y[flagged], rng),
            "recall_ci95": _bootstrap_rate(flagged[y == 1].astype(np.int8), rng),
        }
    )


def fit(
    train: Split, val: Split, params: dict[str, str | int | float] | None = None
) -> xgb.Booster:
    p = dict(DEFAULT_PARAMS if params is None else params)
    negatives, positives = train.rows - train.positives, train.positives
    if positives == 0:
        raise ValueError("training split has no fraud rows")
    p.setdefault("scale_pos_weight", negatives / positives)
    dtrain, dval = make_dmatrix(train.x, train.y), make_dmatrix(val.x, val.y)
    return xgb.train(
        p,
        dtrain,
        num_boost_round=int(p.pop("num_boost_round", MAX_ROUNDS)),
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
    )


def predict(booster: xgb.Booster, split: Split) -> npt.NDArray[np.float64]:
    best = booster.best_iteration
    return np.asarray(
        booster.predict(make_dmatrix(split.x), iteration_range=(0, best + 1)), dtype=np.float64
    )


def train_and_save(
    splits: dict[str, Split],
    model_dir: Path,
    *,
    target_precision: float,
    data_source: str,
    params: dict[str, str | int | float] | None = None,
) -> ModelMetadata:
    used = dict(DEFAULT_PARAMS if params is None else params)
    booster = fit(splits["train"], splits["val"], used)
    val_scores = predict(booster, splits["val"])
    threshold = choose_threshold(splits["val"].y, val_scores, target_precision)

    metrics = {
        "train": evaluate(splits["train"].y, predict(booster, splits["train"]), threshold.value),
        "val": evaluate(splits["val"].y, val_scores, threshold.value),
        "test": evaluate(
            splits["test"].y, predict(booster, splits["test"]), threshold.value, with_ci=True
        ),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(model_dir / MODEL_FILE, bytes(booster.save_raw(raw_format="ubj")))
    sha = sha256_file(model_dir / MODEL_FILE)
    metadata = ModelMetadata(
        model_version=f"fraud_xgb-{sha[:12]}",
        artifact_sha256=sha,
        feature_names=FEATURE_NAMES,
        feature_types=FEATURE_TYPES,
        category_vocab=CATEGORIES,
        threshold=threshold.value,
        threshold_rule=threshold.rule,
        target_precision=target_precision,
        best_iteration=booster.best_iteration,
        trained_at=datetime.now(UTC),
        xgboost_version=xgb.__version__,
        data_source=data_source,
        splits={
            name: f"{s.start.isoformat()} .. {s.end.isoformat()} "
            f"({s.rows} rows, {s.positives} fraud)"
            for name, s in splits.items()
        },
        metrics=metrics,
        params={
            **used,
            "scale_pos_weight": round(
                (splits["train"].rows - splits["train"].positives) / splits["train"].positives, 4
            ),
        },
    )
    write_atomic(model_dir / METADATA_FILE, metadata.model_dump_json(indent=2).encode())
    log.info("model saved", extra={"model_version": metadata.model_version})
    return metadata
