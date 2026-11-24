"""Model API: scores batches of feature vectors with the trained XGBoost model.

POST /v1/score/batch   up to 512 feature vectors -> scores, flags   (X-API-Key)
GET  /v1/model         model metadata                               (X-API-Key)
GET  /healthz          process is alive
GET  /readyz           model loaded
GET  /metrics          Prometheus
"""

# No `from __future__ import annotations`: FastAPI must resolve the closure-local
# dependencies (require_key, loaded_model) in the Annotated[...] hints at runtime.
import hmac
import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import orjson
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from fraud_common.features import N_FEATURES
from fraud_common.logging_config import configure_logging
from fraud_common.model import FraudModel, ModelError, ModelMetadata
from fraud_common.schemas import ScoreBatchRequest, ScoreBatchResponse
from fraud_common.settings import LoggingSettings, ModelApiSettings

log = logging.getLogger(__name__)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "model_api_requests_total", "Requests by route and status", ["route", "status"],
            registry=self.registry,
        )  # fmt: skip
        self.scored = Counter("model_api_scored_total", "Rows scored", registry=self.registry)
        self.flagged = Counter("model_api_flagged_total", "Rows flagged", registry=self.registry)
        self.latency = Histogram(
            "model_api_batch_seconds", "Batch scoring time inside the API",
            buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
            registry=self.registry,
        )  # fmt: skip
        self.batch_size = Histogram(
            "model_api_batch_size", "Rows per batch", buckets=(1, 8, 32, 64, 128, 256, 512),
            registry=self.registry,
        )  # fmt: skip


def _error(status_code: int, code: str, message: str, details: Any = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


def score_rows(current: FraudModel, body: ScoreBatchRequest, metrics: Metrics) -> bytes:
    """Validate shapes, predict, and serialize the response (runs in a worker thread)."""
    started = time.perf_counter()
    for i, item in enumerate(body.items):
        if len(item.features) != N_FEATURES:
            raise RequestValidationError(
                [{"loc": ("body", "items", i, "features"),
                  "msg": f"expected {N_FEATURES} features, got {len(item.features)}"}]
            )  # fmt: skip
    x = np.array(
        [[math.nan if v is None else v for v in item.features] for item in body.items],
        dtype=np.float32,
    )
    if np.isinf(x).any():
        row = int(np.flatnonzero(np.isinf(x).any(axis=1))[0])
        raise RequestValidationError(
            [{"loc": ("body", "items", row, "features"), "msg": "infinite value"}]
        )
    scores = current.predict(x)
    flags = current.flag(scores)
    out = orjson.dumps(
        {
            "model_version": current.version,
            "threshold": current.threshold,
            "results": [
                {"tx_id": item.tx_id, "score": float(sc), "is_flagged": bool(fl)}
                for item, sc, fl in zip(body.items, scores, flags, strict=True)
            ],
        }
    )
    metrics.latency.observe(time.perf_counter() - started)
    metrics.batch_size.observe(len(body.items))
    metrics.scored.inc(len(body.items))
    metrics.flagged.inc(int(flags.sum()))
    return out


def create_app(settings: ModelApiSettings, model: FraudModel | None = None) -> FastAPI:
    """`model` may be injected (tests); otherwise it is loaded from settings.model_dir."""
    metrics = Metrics()
    api_key = settings.api_key.get_secret_value().encode()
    # Largest legitimate body: max_batch rows of 24 long floats, with JSON overhead.
    max_body = 1024 + settings.max_batch * (64 + N_FEATURES * 26)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = model or FraudModel.load(Path(settings.model_dir))
        log.info("model loaded", extra={"model_version": app.state.model.version})
        yield

    app = FastAPI(
        title="fraud-stream model API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.host_list))
    app.state.model = model

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        # Constant-time comparison so response timing doesn't leak the key.
        if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), api_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid X-API-Key")

    def loaded_model(request: Request) -> FraudModel:
        current: FraudModel | None = request.app.state.model
        if current is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
        return current

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        metrics.requests.labels(request.url.path, str(exc.status_code)).inc()
        code = {401: "unauthorized", 413: "batch_too_large", 503: "not_ready"}.get(
            exc.status_code, "http_error"
        )
        return _error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        metrics.requests.labels(request.url.path, "422").inc()
        details = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", "")} for e in exc.errors()[:20]
        ]
        return _error(422, "validation_error", "request body is invalid", details)

    @app.post(
        "/v1/score/batch",
        response_model=ScoreBatchResponse,
        dependencies=[Depends(require_key)],
        summary="Score up to max_batch feature vectors",
        description="Body: ScoreBatchRequest JSON (see schemas). Results come back in order.",
    )
    async def score_batch(
        request: Request, current: Annotated[FraudModel, Depends(loaded_model)]
    ) -> Response:
        # The body is parsed straight from bytes by pydantic-core (faster than FastAPI's
        # json.loads + validate), and the response is serialized once with orjson.
        raw = await request.body()
        if len(raw) > max_body:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"body over {max_body} bytes")
        try:
            body = ScoreBatchRequest.model_validate_json(raw)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from None
        if len(body.items) > settings.max_batch:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE, f"at most {settings.max_batch} items"
            )
        # XGBoost releases the GIL; a worker thread keeps the event loop free meanwhile.
        payload = await run_in_threadpool(score_rows, current, body, metrics)
        metrics.requests.labels("/v1/score/batch", "200").inc()
        return Response(payload, media_type="application/json")

    @app.get("/v1/model", response_model=ModelMetadata, dependencies=[Depends(require_key)])
    def model_info(current: Annotated[FraudModel, Depends(loaded_model)]) -> ModelMetadata:
        return current.metadata

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(current: Annotated[FraudModel, Depends(loaded_model)]) -> dict[str, str]:
        return {"status": "ready", "model_version": current.version}

    @app.get("/metrics")
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    import uvicorn  # noqa: PLC0415  (only needed when run as a server)

    configure_logging(LoggingSettings(), service="model-api")
    settings = ModelApiSettings()
    try:
        model = FraudModel.load(Path(settings.model_dir))
        model.set_threads(settings.threads)
    except ModelError:
        log.exception("refusing to start: model artifact failed verification")
        raise SystemExit(1) from None
    uvicorn.run(
        create_app(settings, model),
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
