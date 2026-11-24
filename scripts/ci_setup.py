"""Prepare a fresh checkout to run the whole stack without the Kaggle download (used by CI).

    uv run python scripts/ci_setup.py env              # .env from .env.example, random secrets
    uv run python scripts/ci_setup.py data OUT.zip     # synthetic Sparkov-like dataset zip

The synthetic data has the same columns and layout as the Kaggle zip, so `make data`,
`make train`, `make up` and the integration tests run unchanged. Its metrics mean nothing.
"""

from __future__ import annotations

import argparse
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = re.compile(r"^([A-Z0-9_]+)=CHANGE_ME\S*$", re.MULTILINE)


def write_env(example: Path, target: Path) -> int:
    """Copy .env.example to .env, replacing every CHANGE_ME value with a random secret."""
    if target.exists():
        raise SystemExit(f"{target} already exists; not overwriting it")
    text, count = PLACEHOLDER.subn(lambda m: f"{m.group(1)}={secrets.token_urlsafe(32)}",
                                   example.read_text(encoding="utf-8"))  # fmt: skip
    target.write_text(text, encoding="utf-8")
    target.chmod(0o600)
    return count


def write_data(out: Path) -> Path:
    sys.path.insert(0, str(ROOT))
    from tests.fixtures.sparkov_like import write_kaggle_zip  # noqa: PLC0415

    return write_kaggle_zip(out, n_cards=60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env")
    data = sub.add_parser("data")
    data.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "env":
        n = write_env(ROOT / ".env.example", ROOT / ".env")
        print(f"wrote .env with {n} generated secrets")
    else:
        print(f"wrote {write_data(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
