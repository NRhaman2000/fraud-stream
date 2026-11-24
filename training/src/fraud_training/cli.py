"""Command line:

fraud-training ingest  --source ~/Downloads/archive.zip --out data/processed
fraud-training train   --data data/processed --models models [--target-precision 0.92]
fraud-training report  --models models
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from fraud_common.features import FeatureEngine
from fraud_common.logging_config import configure_logging
from fraud_common.model import FraudModel, ModelError, ModelMetadata
from fraud_common.settings import LoggingSettings
from fraud_training.dataset import build_features, load_profiles, split
from fraud_training.ingest import IngestError, ingest
from fraud_training.train import train_and_save


def format_report(meta: ModelMetadata) -> str:
    lines = [
        f"model            {meta.model_version}",
        f"threshold        {meta.threshold:.4f}  ({meta.threshold_rule})",
        f"best iteration   {meta.best_iteration}",
        "",
        f"{'split':<6} {'rows':>9} {'fraud':>6} {'precision':>10} {'recall':>8} "
        f"{'f1':>6} {'PR-AUC':>7} {'ROC-AUC':>8} {'TP':>6} {'FP':>6} {'FN':>6}",
    ]
    for name, m in meta.metrics.items():
        lines.append(
            f"{name:<6} {m.rows:>9,} {m.positives:>6,} {m.precision:>10.3f} {m.recall:>8.3f} "
            f"{m.f1:>6.3f} {m.pr_auc:>7.3f} {m.roc_auc:>8.3f} {m.tp:>6,} {m.fp:>6,} {m.fn:>6,}"
        )
    test = meta.metrics["test"]
    if test.precision_ci95 and test.recall_ci95:
        lines += [
            "",
            f"test precision {test.precision:.3f} (95% CI {test.precision_ci95[0]:.3f}-"
            f"{test.precision_ci95[1]:.3f}), recall {test.recall:.3f} "
            f"(95% CI {test.recall_ci95[0]:.3f}-{test.recall_ci95[1]:.3f})",
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraud-training")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ing = sub.add_parser("ingest", help="clean the Kaggle CSVs into parquet")
    p_ing.add_argument("--source", type=Path, required=True, help="zip or folder with both CSVs")
    p_ing.add_argument("--out", type=Path, default=Path("data/processed"))
    p_tr = sub.add_parser("train", help="build features, train, evaluate, save artifact")
    p_tr.add_argument("--data", type=Path, default=Path("data/processed"))
    p_tr.add_argument("--models", type=Path, default=Path("models"))
    p_tr.add_argument("--target-precision", type=float, default=0.95)
    p_rep = sub.add_parser("report", help="print metrics of the saved model")
    p_rep.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args(argv)

    configure_logging(LoggingSettings(format="console"), service="training")
    try:
        if args.command == "ingest":
            r = ingest(args.source.expanduser(), args.out)
            print(
                f"rows in {r.rows_in:,} -> out {r.rows_out:,} (invalid {r.dropped_invalid:,}, "
                f"duplicate {r.dropped_duplicate:,}); cards {r.cards:,}; "
                f"profile conflicts {r.profile_conflicts}; fraud rate {r.fraud_rate:.4%}"
            )
        elif args.command == "train":
            started = time.perf_counter()
            if not (args.data / "transactions.parquet").is_file():
                print(
                    f"error: no cleaned data in {args.data}/ yet. Run first:  "
                    "make data SRC=path/to/kaggle-archive.zip",
                    file=sys.stderr,
                )
                return 1
            tx = pd.read_parquet(args.data / "transactions.parquet")
            profiles = load_profiles(pd.read_parquet(args.data / "profiles.parquet"))
            x = build_features(tx, FeatureEngine(profiles))
            elapsed = time.perf_counter() - started
            print(f"features: {x.shape[0]:,} rows x {x.shape[1]} in {elapsed:.1f}s")
            meta = train_and_save(
                split(tx, x),
                args.models,
                target_precision=args.target_precision,
                data_source="kaggle:kartik2112/fraud-detection",
            )
            FraudModel.load(args.models)  # prove the artifact round-trips
            print(format_report(meta))
            print(f"total {time.perf_counter() - started:.1f}s")
        else:
            print(format_report(FraudModel.load(args.models).metadata))
    except (IngestError, ModelError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
