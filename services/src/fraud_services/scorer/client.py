"""HTTP client for the model API with bounded, jittered retries."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable

import httpx

from fraud_common.schemas import ScoreBatchRequest, ScoreBatchResponse

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """The model API could not be reached or answered 5xx/429. Safe to retry."""


class ModelRejectedError(RuntimeError):
    """The model API refused the request (4xx). Retrying the same batch won't help."""


class ModelClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            timeout=timeout_s,
            transport=transport,
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )

    def close(self) -> None:
        self._http.close()

    def score(self, request: ScoreBatchRequest) -> ScoreBatchResponse:
        try:
            resp = self._http.post("/v1/score/batch", content=request.model_dump_json())
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code == 200:
            return ScoreBatchResponse.model_validate_json(resp.content)
        detail = resp.text[:500]
        if resp.status_code == 429 or resp.status_code >= 500:
            raise ModelUnavailableError(f"HTTP {resp.status_code}: {detail}")
        raise ModelRejectedError(f"HTTP {resp.status_code}: {detail}")


class Backoff:
    """Exponential backoff with full jitter: 50 ms, 100 ms, ... capped at `max_s`."""

    def __init__(
        self,
        max_s: float,
        base_s: float = 0.05,
        rand: Callable[[], float] = random.random,
    ) -> None:
        if base_s <= 0 or max_s < base_s:
            raise ValueError("need 0 < base_s <= max_s")
        self._base, self._max, self._rand = base_s, max_s, rand
        self.attempt = 0

    def next_delay(self) -> float:
        cap = min(self._max, self._base * 2**self.attempt)
        self.attempt += 1
        return float(max(self._base / 2, cap * self._rand()))

    def reset(self) -> None:
        self.attempt = 0


def score_with_retry(  # noqa: PLR0913  (injectable sleep/callbacks for tests)
    client: ModelClient,
    request: ScoreBatchRequest,
    backoff: Backoff,
    keep_alive: Callable[[], None],
    *,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[str], None] | None = None,
) -> ScoreBatchResponse:
    """Retry until the model answers. `keep_alive` runs between attempts (the scorer uses it
    to keep its Kafka group membership while consumption is paused)."""
    while True:
        try:
            response = client.score(request)
        except ModelUnavailableError as exc:
            delay = backoff.next_delay()
            if on_retry:
                on_retry(str(exc))
            log.warning(
                "model API unavailable; retrying",
                extra={"error": str(exc), "delay_s": round(delay, 3)},
            )
            sleep(delay)
            keep_alive()
            continue
        backoff.reset()
        return response
