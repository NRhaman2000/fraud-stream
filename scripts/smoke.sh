#!/usr/bin/env bash
# Gate 2 smoke test: the empty shell builds, starts, and every piece is reachable.
# SMOKE_SKIP_BUILD=1 reuses an image built earlier (e.g. by a separate CI step).
set -euo pipefail
cd "$(dirname "$0")/.."

pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; exit 1; }
# step "<description>" cmd args...  -> PASS, or FAIL and stop (works under set -e)
step() { local d="$1"; shift; if "$@"; then pass "$d"; else fail "$d"; fi; }

[[ -f .env ]] || fail ".env exists (cp .env.example .env)"
if grep -q 'CHANGE_ME' .env; then
  echo "  WARN .env still has CHANGE_ME values; fine for a local smoke test, not for anything else"
fi

echo "1. Build image"
if [[ "${SMOKE_SKIP_BUILD:-0}" == "1" ]]; then
  step "using prebuilt image fraud-stream/app:dev" docker image inspect -f '{{.Id}}' fraud-stream/app:dev
else
  step "image fraud-stream/app:dev built" docker compose build --quiet
fi

echo "2. Start stack and wait for healthchecks"
step "kafka, postgres, prometheus, grafana healthy" \
  docker compose up -d --wait --wait-timeout 180 kafka postgres prometheus grafana

echo "3. Create topics (one-shot job)"
log_file="$(mktemp)"
docker compose up kafka-init --exit-code-from kafka-init --no-log-prefix >"$log_file" 2>&1 \
  || { cat "$log_file"; fail "kafka-init exited non-zero"; }
pass "kafka-init exited 0"

echo "4. Verify topics + Postgres schema from inside the network"
step "fraud_common.admin check exited 0" docker compose run --rm --no-deps smoke

echo "5. Container hardening"
uid="$(docker compose run --rm --no-deps --entrypoint id smoke -u | tr -d '\r')"
step "app container runs as uid 10001 (non-root, got $uid)" test "$uid" = "10001"

echo "6. Host ports bound to localhost only"
if docker compose ps --format '{{.Ports}}' | grep -q '0\.0\.0\.0:'; then
  fail "a port is exposed on 0.0.0.0"
fi
pass "all published ports are 127.0.0.1"

echo "7. HTTP endpoints from the host"
step "prometheus ready" curl -fsS -o /dev/null http://127.0.0.1:9090/-/ready
step "grafana healthy" bash -c "curl -fsS http://127.0.0.1:3000/api/health | grep -q '\"database\": *\"ok\"'"
step "prometheus scraping itself" bash -c "curl -fsS http://127.0.0.1:9090/api/v1/targets | grep -q '\"health\":\"up\"'"

echo
echo "Smoke test passed."
