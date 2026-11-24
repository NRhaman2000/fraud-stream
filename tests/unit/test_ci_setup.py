import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ci_setup", ROOT / "scripts" / "ci_setup.py")
assert spec is not None
assert spec.loader is not None
ci_setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_setup)


def test_env_from_example_has_no_placeholders(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    count = ci_setup.write_env(ROOT / ".env.example", target)
    text = target.read_text(encoding="utf-8")
    assert count >= 4
    assert "CHANGE_ME" not in text
    assert target.stat().st_mode & 0o777 == 0o600
    keys = [line.split("=", 1)[0] for line in text.splitlines() if "=" in line and line[0] != "#"]
    assert {"POSTGRES_PASSWORD", "POSTGRES_APP_PASSWORD", "MODEL_API_API_KEY"} <= set(keys)
    with pytest.raises(SystemExit, match="already exists"):
        ci_setup.write_env(ROOT / ".env.example", target)


def test_synthetic_dataset_zip(tmp_path: Path) -> None:
    out = ci_setup.write_data(tmp_path / "archive.zip")
    assert out.is_file()
    assert out.stat().st_size > 10_000
