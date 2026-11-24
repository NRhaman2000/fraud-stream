"""Dashboard: live stream stats and the analyst alert queue.

GET   /                     the dashboard page
GET   /api/stream           Server-Sent Events: `stats` every push interval, `alert` per flag
GET   /api/stats            one stats snapshot
GET   /api/alerts           review queue, newest first, keyset-paginated
PATCH /api/alerts/{tx_id}   analyst decision: confirmed_fraud | false_positive
GET   /healthz /readyz /metrics
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Protocol

import orjson
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, generate_latest
from psycopg_pool import ConnectionPool
from pydantic import AwareDatetime, BaseModel, ConfigDict
from starlette.middleware.trustedhost import TrustedHostMiddleware

from fraud_common.alerts import (
    MAX_PAGE,
    Alert,
    AlertAlreadyReviewedError,
    AlertNotFoundError,
    AlertStatus,
    PageCursor,
    list_alerts,
    review_alert,
)
from fraud_common.logging_config import configure_logging
from fraud_common.settings import (
    DashboardSettings,
    KafkaSettings,
    LoggingSettings,
    PostgresSettings,
)
from fraud_services.dashboard.stats import LiveStats

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
TX_ID_PATTERN = r"^[0-9a-f]{32}$"


class AlertStore(Protocol):
    def list(
        self, status: AlertStatus | None, limit: int, before: PageCursor | None
    ) -> list[Alert]: ...
    def review(self, tx_id: str, decision: AlertStatus) -> Alert: ...
    def ping(self) -> bool: ...


class PostgresAlertStore:
    def __init__(self, settings: PostgresSettings) -> None:
        self._pool = ConnectionPool(
            kwargs=settings.app_connect_kwargs(),
            min_size=1,
            max_size=4,
            open=False,
            name="dashboard",
        )

    def open(self) -> None:
        self._pool.open(wait=False)

    def close(self) -> None:
        self._pool.close()

    def list(
        self, status: AlertStatus | None, limit: int, before: PageCursor | None
    ) -> list[Alert]:
        with self._pool.connection() as conn:
            return list_alerts(conn, status=status, limit=limit, before=before)

    def review(self, tx_id: str, decision: AlertStatus) -> Alert:
        with self._pool.connection() as conn:
            return review_alert(conn, tx_id, decision)

    def ping(self) -> bool:
        try:
            with self._pool.connection(timeout=2) as conn:
                conn.execute("SELECT 1")
        except Exception:  # readiness probe: any failure means not ready
            return False
        return True


class ReviewBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: AlertStatus


class AlertPage(BaseModel):
    items: list[Alert]
    next_before_flagged_at: AwareDatetime | None
    next_before_tx_id: str | None


def _error(code: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={"error": {"code": kind, "message": message}})


def sse(event: str, data: Any) -> bytes:
    return b"event: " + event.encode() + b"\ndata: " + orjson.dumps(data) + b"\n\n"


def create_app(  # noqa: PLR0915  (route definitions)
    settings: DashboardSettings,
    stats: LiveStats,
    store: AlertStore,
    consumer_alive: Any = None,
    lifespan_hooks: tuple[Any, Any] | None = None,
) -> FastAPI:
    """`consumer_alive` is a callable -> bool; `lifespan_hooks` = (startup, shutdown)."""
    registry = CollectorRegistry()
    reviews = Counter("dashboard_reviews_total", "Analyst reviews", ["decision"], registry=registry)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if lifespan_hooks:
            lifespan_hooks[0]()
        yield
        if lifespan_hooks:
            lifespan_hooks[1]()

    app = FastAPI(title="fraud-stream dashboard", version="1.0.0", lifespan=lifespan,
                  docs_url="/docs", redoc_url=None)  # fmt: skip
    # Only answer requests addressed to us by name: a web page that re-points its own
    # domain at 127.0.0.1 (DNS rebinding) sends its own Host header and is refused.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.host_list))
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(HTTPException)
    async def http_error(_r: Request, exc: HTTPException) -> JSONResponse:
        kind = {404: "not_found", 409: "already_reviewed", 503: "not_ready"}.get(
            exc.status_code, "http_error"
        )
        return _error(exc.status_code, kind, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_r: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()))
        return _error(422, "validation_error", f"{where}: {first.get('msg', 'invalid')}")

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/stats")
    def get_stats() -> dict[str, object]:
        return stats.snapshot().as_dict()

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        async def events() -> AsyncIterator[bytes]:
            seq = max(0, stats.last_seq - 20)  # start with the last 20 alerts
            yield b"retry: 2000\n\n"
            while not await request.is_disconnected():
                for note in stats.alerts_after(seq):
                    seq = note.seq
                    yield sse("alert", note.event.model_dump(mode="json"))
                yield sse("stats", stats.snapshot().as_dict())
                await asyncio.sleep(settings.push_interval_s)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/alerts", response_model=AlertPage)
    def get_alerts(
        status_: Annotated[AlertStatus | None, Query(alias="status")] = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 50,
        before_flagged_at: Annotated[datetime | None, Query()] = None,
        before_tx_id: Annotated[str | None, Query(pattern=TX_ID_PATTERN)] = None,
    ) -> AlertPage:
        if (before_flagged_at is None) != (before_tx_id is None):
            raise HTTPException(422, "before_flagged_at and before_tx_id go together")
        cursor = None
        if before_flagged_at is not None and before_tx_id is not None:
            if before_flagged_at.tzinfo is None:
                raise HTTPException(422, "before_flagged_at must include a timezone")
            cursor = PageCursor(flagged_at=before_flagged_at, tx_id=before_tx_id)
        items = store.list(status_, limit, cursor)
        last = items[-1] if len(items) == limit else None
        return AlertPage(
            items=items,
            next_before_flagged_at=last.flagged_at if last else None,
            next_before_tx_id=last.tx_id if last else None,
        )

    @app.patch("/api/alerts/{tx_id}", response_model=Alert)
    def patch_alert(
        tx_id: Annotated[str, PathParam(pattern=TX_ID_PATTERN)], body: ReviewBody
    ) -> Alert:
        if body.status is AlertStatus.OPEN:
            raise HTTPException(422, "status must be confirmed_fraud or false_positive")
        try:
            alert = store.review(tx_id, body.status)
        except AlertNotFoundError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no alert with that tx_id") from None
        except AlertAlreadyReviewedError:
            raise HTTPException(status.HTTP_409_CONFLICT, "alert was already reviewed") from None
        reviews.labels(body.status.value).inc()
        return alert

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        if consumer_alive is not None and not consumer_alive():
            raise HTTPException(503, "Kafka consumer is not running")
        if not store.ping():
            raise HTTPException(503, "database unreachable")
        return {"status": "ready"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


def main() -> None:
    import uvicorn  # noqa: PLC0415

    from fraud_services.dashboard.consumer import DashboardConsumer  # noqa: PLC0415

    configure_logging(LoggingSettings(), service="dashboard")
    settings = DashboardSettings()
    postgres = PostgresSettings()
    stats = LiveStats(settings.stats_window_s)
    store = PostgresAlertStore(postgres)
    consumer = DashboardConsumer(stats, settings, KafkaSettings(), postgres)

    def startup() -> None:
        store.open()
        consumer.start()

    def shutdown() -> None:
        consumer.stop()
        consumer.join(5)
        store.close()

    app = create_app(settings, stats, store, consumer.healthy.is_set, (startup, shutdown))
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
