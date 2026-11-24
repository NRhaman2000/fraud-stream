# syntax=docker/dockerfile:1.7
# One image for every Python service; each compose service picks its own command.
# Only runtime packages go in: the offline training pipeline (pandas, scikit-learn) stays out.
#
# Behind a TLS-intercepting corporate proxy, pass its CA without baking it into a layer:
#   docker build --secret id=extra_ca,src=/path/to/ca.pem ...

FROM python:3.12-slim AS builder
ARG UV_VERSION=0.11.32
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    pip install --no-cache-dir --root-user-action=ignore "uv==${UV_VERSION}"
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# Dependencies first, so code edits don't invalidate the dependency layer.
COPY pyproject.toml uv.lock ./
COPY libs/fraud_common/pyproject.toml libs/fraud_common/pyproject.toml
COPY training/pyproject.toml training/pyproject.toml
COPY services/pyproject.toml services/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --frozen --no-dev --package fraud-services --no-install-workspace
COPY libs ./libs
COPY services ./services
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --frozen --no-dev --package fraud-services --no-editable

FROM python:3.12-slim AS runtime
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
USER app
CMD ["python", "-m", "fraud_common.admin", "check"]
