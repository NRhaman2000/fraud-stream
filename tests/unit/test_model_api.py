from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fraud_common.features import FEATURE_NAMES, N_FEATURES
from fraud_common.model import FraudModel
from fraud_common.settings import ModelApiSettings
from fraud_services.model_api.app import create_app
from tests.fixtures.model_artifact import write_artifact

KEY = "k" * 40
TX = "ab" * 16


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> FraudModel:
    return FraudModel.load(write_artifact(tmp_path_factory.mktemp("model")))


def _settings(**kw: Any) -> ModelApiSettings:
    return ModelApiSettings(api_key=KEY, **kw)


@pytest.fixture
def client(model: FraudModel) -> Iterator[TestClient]:
    with TestClient(create_app(_settings(max_batch=4), model), base_url="http://localhost") as c:
        yield c


def item(amount: float = 0.0, tx: str = TX, n: int = N_FEATURES) -> dict[str, Any]:
    features: list[float | None] = [0.0] * n
    if n:
        features[0] = amount
    if n > 7:
        features[7] = None  # missing value is allowed
    return {"tx_id": tx, "features": features}


def post(client: TestClient, items: list[dict[str, Any]], key: str | None = KEY) -> Any:
    headers = {"X-API-Key": key} if key is not None else {}
    return client.post("/v1/score/batch", json={"items": items}, headers=headers)


def test_scores_batch_in_order(client: TestClient, model: FraudModel) -> None:
    resp = post(client, [item(5.0, "a" * 32), item(-5.0, "b" * 32)])
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == model.version
    assert body["threshold"] == 0.5
    assert [r["tx_id"] for r in body["results"]] == ["a" * 32, "b" * 32]
    high, low = body["results"]
    assert high["score"] > low["score"]
    assert high["is_flagged"] is True
    assert low["is_flagged"] is False


@pytest.mark.parametrize("key", [None, "", "wrong", KEY + "x", KEY[:-1]])
def test_rejects_bad_api_key(client: TestClient, key: str | None) -> None:
    resp = post(client, [item()], key=key)
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_model_info_requires_key(client: TestClient) -> None:
    assert client.get("/v1/model").status_code == 401
    info = client.get("/v1/model", headers={"X-API-Key": KEY}).json()
    assert info["feature_names"] == list(FEATURE_NAMES)


@pytest.mark.parametrize(
    "body",
    [
        {"items": []},
        {"items": [item(n=N_FEATURES - 1)]},
        {"items": [item(n=N_FEATURES + 1)]},
        {"items": [{"tx_id": "nothex", "features": [0.0] * N_FEATURES}]},
        {"items": [{**item(), "extra": 1}]},
        {"items": [{"tx_id": TX, "features": ["a"] * N_FEATURES}]},
        {},
    ],
)
def test_rejects_invalid_bodies(client: TestClient, body: dict[str, Any]) -> None:
    resp = client.post("/v1/score/batch", json=body, headers={"X-API-Key": KEY})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_rejects_infinite_feature(client: TestClient) -> None:
    zeros = ",0" * (N_FEATURES - 1)
    raw = '{"items":[{"tx_id":"' + TX + '","features":[1e400' + zeros + "]}]}"
    resp = client.post(
        "/v1/score/batch",
        content=raw,
        headers={"X-API-Key": KEY, "Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_rejects_oversized_batch(client: TestClient) -> None:
    resp = post(client, [item()] * 5)  # settings.max_batch = 4
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "batch_too_large"


def test_rejects_malformed_json(client: TestClient) -> None:
    resp = client.post(
        "/v1/score/batch",
        content=b"{not json",
        headers={"X-API-Key": KEY, "Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_health_ready_metrics(client: TestClient, model: FraudModel) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["model_version"] == model.version
    post(client, [item()])
    text = client.get("/metrics").text
    assert "model_api_scored_total 1.0" in text
    assert "model_api_batch_seconds_bucket" in text


def test_loads_model_from_disk(tmp_path: Path) -> None:
    write_artifact(tmp_path)
    with TestClient(
        create_app(_settings(model_dir=str(tmp_path))), base_url="http://localhost"
    ) as c:
        assert c.get("/readyz").status_code == 200


def test_refuses_to_start_with_tampered_model(tmp_path: Path) -> None:
    write_artifact(tmp_path)
    with (tmp_path / "model.ubj").open("ab") as fh:
        fh.write(b"x")
    from fraud_common.model import ModelError  # noqa: PLC0415

    with (
        pytest.raises(ModelError, match="sha256"),
        TestClient(create_app(_settings(model_dir=str(tmp_path))), base_url="http://localhost"),
    ):
        pass


def test_foreign_host_header_is_refused(client: TestClient) -> None:
    resp = client.get("/healthz", headers={"Host": "evil.example"})
    assert resp.status_code == 400
    assert client.get("/healthz", headers={"Host": "model-api:8001"}).status_code == 200


def test_oversized_body_is_refused(client: TestClient) -> None:
    resp = client.post(
        "/v1/score/batch",
        content=b"[" + b" " * 300_000 + b"]",
        headers={"X-API-Key": KEY, "Content-Type": "application/json"},
    )
    assert resp.status_code == 413
