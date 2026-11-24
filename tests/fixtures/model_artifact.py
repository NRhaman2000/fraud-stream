"""A tiny but real XGBoost artifact with the production feature contract, for tests."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

from fraud_common.features import CATEGORIES, FEATURE_NAMES, FEATURE_TYPES, N_FEATURES
from fraud_common.model import (
    METADATA_FILE,
    MODEL_FILE,
    ModelMetadata,
    SplitMetrics,
    make_dmatrix,
    sha256_file,
    write_atomic,
)

ZERO_METRICS = SplitMetrics(
    rows=0, positives=0, precision=0, recall=0, f1=0, pr_auc=0, roc_auc=0, tp=0, fp=0, fn=0, tn=0
)


def write_artifact(path: Path, threshold: float = 0.5) -> Path:
    """Model where feature 0 (amount) > 0.8 means fraud."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, N_FEATURES)).astype(np.float32)
    x[:, FEATURE_NAMES.index("category")] = rng.integers(0, 15, 400)
    y = (x[:, 0] > 0.8).astype(np.int8)
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 3, "seed": 0},
        make_dmatrix(x, y),
        num_boost_round=10,
    )
    write_atomic(path / MODEL_FILE, bytes(booster.save_raw(raw_format="ubj")))
    sha = sha256_file(path / MODEL_FILE)
    meta = ModelMetadata(
        model_version=f"fraud_xgb-{sha[:12]}",
        artifact_sha256=sha,
        feature_names=FEATURE_NAMES,
        feature_types=FEATURE_TYPES,
        category_vocab=CATEGORIES,
        threshold=threshold,
        threshold_rule="test",
        target_precision=0.95,
        best_iteration=9,
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        xgboost_version=xgb.__version__,
        data_source="unit-test",
        splits={},
        metrics={"test": ZERO_METRICS},
        params={},
    )
    (path / METADATA_FILE).write_text(meta.model_dump_json())
    return path
