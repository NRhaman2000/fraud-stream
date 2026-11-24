import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Unit tests must not read the developer's real .env or exported variables."""
    prefixes = ("LOG_", "KAFKA_", "POSTGRES_", "PRODUCER_", "SCORER_", "MODEL_API_", "DASHBOARD_")
    for key in list(os.environ):
        if key.startswith(prefixes):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
