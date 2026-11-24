"""Benchmark: stream at a fixed rate, then read throughput and latency from Prometheus.

    uv run python scripts/bench.py --rate 5000 --seconds 60

Latency is end to end per transaction: from the moment the producer hands it to Kafka
until the scorer has its score (scorer_e2e_latency_seconds). The first `--warmup`
seconds and the drain at the end are left out of the measured window.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass

PROM = "http://localhost:9090/api/v1/query"
MIN_WINDOW_S = 10


def prom(query: str, at: float) -> float | None:
    url = f"{PROM}?{urllib.parse.urlencode({'query': query, 'time': f'{at:.3f}'})}"
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (fixed local URL)
        result = json.load(resp)["data"]["result"]
    if not result:
        return None
    value = float(result[0]["value"][1])
    return None if math.isnan(value) else value


def scored_total(at: float) -> float:
    return prom("sum(scorer_scored_total)", at) or 0.0


def scored_sample(at: float) -> tuple[float, float]:
    """(scored count, time Prometheus actually scraped it): the newest raw sample at or
    before `at`. Using the scrape time instead of the query time keeps the throughput
    exact despite the 5 s scrape interval."""
    params = {"query": "scorer_scored_total[30s]", "time": f"{at:.3f}"}
    url = f"{PROM}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (fixed local URL)
        result = json.load(resp)["data"]["result"]
    samples = [(float(ts), float(v)) for series in result for ts, v in series["values"]]
    if not samples:
        raise SystemExit("no scorer metrics in Prometheus; is the stack up (make up)?")
    scraped, value = max(samples)
    return value, scraped


@dataclass(frozen=True)
class Result:
    target_tps: int
    achieved_tps: float
    window_s: float
    scored: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    kafka_p50_ms: float | None  # produced -> read by the scorer (part of the total)
    kafka_p95_ms: float | None
    model_call_p95_ms: float | None
    avg_batch: float | None
    dead_letters: int


def run_producer(rate: int, seconds: int) -> subprocess.Popen[bytes]:
    cmd = [
        "docker", "compose", "--profile", "producer", "run", "--rm",
        "-e", f"PRODUCER_RATE_TPS={rate}",
        "-e", f"PRODUCER_MAX_EVENTS={rate * seconds}",
        "-e", "PRODUCER_LOOP=true",
        # Labels go out right away, so the producer stops as soon as the last
        # transaction is sent (otherwise its label wait would count as idle time).
        "-e", "PRODUCER_LABEL_DELAY_S=0",
        "producer",
    ]  # fmt: skip
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603


def measure(rate: int, seconds: int, warmup: int) -> Result:
    if seconds <= warmup + 10:
        raise SystemExit("--seconds must be at least warmup + 10")
    before = scored_total(time.time())
    proc = run_producer(rate, seconds)
    # Wait for the first scored events (the producer loads its data first).
    started = None
    deadline = time.time() + 120
    while started is None and time.time() < deadline:
        time.sleep(1)
        if scored_total(time.time()) > before:
            started = time.time()
    if started is None:
        proc.kill()
        raise SystemExit("nothing was scored within 120 s; is the stack up (make up)?")
    if proc.wait() != 0:
        raise SystemExit("the producer failed; see `docker compose logs producer`")
    end = time.time() - 3  # leave out the final drain
    time.sleep(6)  # let Prometheus scrape past `end` (scrape interval 5 s)
    begin = started + warmup
    n0, t0 = scored_sample(begin)
    n1, t1 = scored_sample(end)
    window = t1 - t0
    if window < MIN_WINDOW_S:
        raise SystemExit(
            f"measured window only {window:.0f} s; did a service restart mid-run? Try again."
        )
    rng = f"[{int(window)}s]"
    scored = n1 - n0

    def q(expr: str) -> float | None:
        v = prom(expr, end)
        return None if v is None else round(v * 1000, 2)

    def quantile(phi: float, metric: str) -> float | None:
        return q(f"histogram_quantile({phi}, sum(increase({metric}_bucket{rng})) by (le))")

    batch_avg = prom(
        f"sum(increase(scorer_batch_size_sum{rng}))/sum(increase(scorer_batch_size_count{rng}))",
        end,
    )
    return Result(
        target_tps=rate,
        achieved_tps=round(scored / window, 1),
        window_s=round(window, 1),
        scored=int(scored),
        p50_ms=quantile(0.50, "scorer_e2e_latency_seconds"),
        p95_ms=quantile(0.95, "scorer_e2e_latency_seconds"),
        p99_ms=quantile(0.99, "scorer_e2e_latency_seconds"),
        kafka_p50_ms=quantile(0.50, "scorer_fetch_latency_seconds"),
        kafka_p95_ms=quantile(0.95, "scorer_fetch_latency_seconds"),
        model_call_p95_ms=quantile(0.95, "scorer_model_call_seconds"),
        avg_batch=None if batch_avg is None else round(batch_avg, 1),
        dead_letters=int(prom(f"sum(increase(scorer_dead_letters_total{rng}))", end) or 0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rate", type=int, nargs="+", default=[5000], help="target tx/s")
    parser.add_argument("--seconds", type=int, default=60, help="producer run time per rate")
    parser.add_argument("--warmup", type=int, default=10, help="seconds ignored at the start")
    args = parser.parse_args(argv)
    results = []
    for rate in args.rate:
        if not 1 <= rate <= 10_000:
            parser.error("--rate must be 1..10000")
        print(f"streaming {rate} tx/s for {args.seconds} s ...", file=sys.stderr, flush=True)
        results.append(measure(rate, args.seconds, args.warmup))
        print(json.dumps(asdict(results[-1])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
