.DEFAULT_GOAL := help
SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: help install lint format typecheck test test-integration check build up down clean smoke data train report demo logs bench audit drill

help: ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

install: ## Create .venv with all dev dependencies (Python 3.12 via uv)
	uv sync --frozen
	uv run pre-commit install

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint and formatting
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## mypy --strict
	uv run mypy

test: ## Unit tests with coverage gate (85%)
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test ## Everything CI runs without Docker

data: ## Clean the Kaggle download:  make data SRC=~/Downloads/archive.zip
	@test -n "$(SRC)" || { echo "usage: make data SRC=path/to/kaggle.zip (or folder with both CSVs)"; exit 1; }
	uv run fraud-training ingest --source "$(SRC)" --out data/processed

train: ## Build features, train, pick threshold, evaluate on test, save models/
	uv run fraud-training train --data data/processed --models models --target-precision 0.95

report: ## Print the saved model's metrics
	uv run fraud-training report --models models

.env:
	@echo ".env missing: run  cp .env.example .env  and replace every CHANGE_ME value" && exit 1

build: .env ## Build the app image
	$(COMPOSE) build

up: .env ## Start everything (needs `make data` and `make train` first)
	@test -f models/model.ubj || { echo "no model yet: run  make train  first"; exit 1; }
	@test -f data/processed/profiles.parquet || { echo "no data yet: run  make data SRC=...  first"; exit 1; }
	@grep -qE '^POSTGRES_APP_PASSWORD=.{12,}' .env || { echo "add POSTGRES_APP_PASSWORD=<12+ characters> to .env (see .env.example)"; exit 1; }
	$(COMPOSE) up -d --build --wait kafka postgres prometheus grafana model-api scorer dashboard
	@echo "Dashboard: http://localhost:8000   Grafana: http://localhost:3000   Model API docs: http://localhost:8001/docs"

RATE ?= 1000
MAX ?= 0
demo: .env ## Stream transactions into the system:  make demo RATE=1000 [MAX=50000]
	PRODUCER_RATE_TPS=$(RATE) PRODUCER_MAX_EVENTS=$(MAX) $(COMPOSE) --profile producer run --rm \
	  -e PRODUCER_RATE_TPS=$(RATE) -e PRODUCER_MAX_EVENTS=$(MAX) producer

SECONDS_PER_RATE ?= 60
bench: .env ## Throughput + latency benchmark on the running stack:  make bench RATES="1000 5000"
	uv run python scripts/bench.py --rate $(or $(RATES),5000) --seconds $(SECONDS_PER_RATE)

drill: .env ## Outage drill: kill/stop scorer, model API and Postgres under load, check nothing is lost
	bash scripts/drill.sh

audit: ## Check runtime dependencies against known vulnerabilities (pip-audit)
	uv export --frozen --no-dev --no-emit-workspace --no-hashes -q -o .audit-requirements.txt
	uvx pip-audit -r .audit-requirements.txt --no-deps --disable-pip --progress-spinner off
	@rm -f .audit-requirements.txt

logs: ## Follow the application logs
	$(COMPOSE) logs -f --tail 50 model-api scorer dashboard

smoke: .env ## Gate 2 smoke test: build, start, verify, report
	bash scripts/smoke.sh

test-integration: .env ## Integration tests against the running stack
	IT_POSTGRES_PASSWORD="$$(grep -E '^POSTGRES_PASSWORD=' .env | cut -d= -f2-)" \
	IT_MODEL_API_KEY="$$(grep -E '^MODEL_API_API_KEY=' .env | cut -d= -f2-)" \
	IT_POSTGRES_APP_PASSWORD="$$(grep -E '^POSTGRES_APP_PASSWORD=' .env | cut -d= -f2-)" \
	  uv run pytest -m integration -p no:cacheprovider

down: ## Stop the stack (keeps data volumes)
	$(COMPOSE) --profile smoke --profile producer down

clean: ## Stop the stack AND delete its data volumes
	$(COMPOSE) --profile smoke --profile producer down --volumes --remove-orphans
