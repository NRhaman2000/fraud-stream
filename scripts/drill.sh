#!/usr/bin/env bash
# Outage drill under live traffic (needs `make up`):
#   60,000 transactions at 1,000/s while the scorer is killed, the model API is stopped
#   and Postgres is stopped. Passes when every transaction is scored exactly once, every
#   flagged one is stored as an alert, and the dashboard is ready again at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE="docker compose"
TOTAL=60000

stats() {
  curl -fsS localhost:8000/api/stats |
    python3 -c "import sys,json; d=json.load(sys.stdin); print(d['scored_total'], d['flagged_total'])"
}
alert_rows() {
  $COMPOSE exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "select count(*) from alerts"'
}
say() { printf '%s  %s\n' "$(date +%T)" "$*"; }

read -r s0 f0 < <(stats)
a0=$(alert_rows)
say "start: scored=$s0 flagged=$f0 alert_rows=$a0"
$COMPOSE --profile producer run --rm -e PRODUCER_RATE_TPS=1000 -e PRODUCER_MAX_EVENTS=$TOTAL \
  -e PRODUCER_LABEL_DELAY_S=2 -e PRODUCER_LOOP=false producer >/dev/null 2>&1 &
producer_pid=$!

sleep 15; say "kill -9 scorer";            $COMPOSE kill -s SIGKILL scorer >/dev/null 2>&1
sleep 3;  say "start scorer";              $COMPOSE start scorer >/dev/null 2>&1
sleep 30; say "stop model-api for 10 s";   $COMPOSE stop model-api >/dev/null 2>&1
sleep 10; say "start model-api";           $COMPOSE start model-api >/dev/null 2>&1
sleep 5;  say "stop postgres for 15 s";    $COMPOSE stop postgres >/dev/null 2>&1
sleep 4;  say "dashboard /readyz while the database is down: $(curl -s -o /dev/null -w '%{http_code}' localhost:8000/readyz) (want 503)"
sleep 11; say "start postgres";            $COMPOSE start postgres >/dev/null 2>&1

wait "$producer_pid"
say "producer finished; waiting for the backlog to drain"
for _ in $(seq 1 120); do
  read -r s _ < <(stats) || true
  [ $((s - s0)) -ge $TOTAL ] && break
  sleep 2
done
sleep 5
read -r s1 f1 < <(stats)
a1=$(alert_rows)
ready=$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/readyz)
say "scored:      $((s1 - s0)) (want $TOTAL)"
say "flagged:     $((f1 - f0))"
say "alert rows:  $((a1 - a0)) (want the same as flagged)"
say "dashboard:   /readyz $ready (want 200)"
if [ $((s1 - s0)) -eq $TOTAL ] && [ $((f1 - f0)) -eq $((a1 - a0)) ] && [ "$ready" = 200 ]; then
  say "DRILL PASSED"
else
  say "DRILL FAILED"
  exit 1
fi
