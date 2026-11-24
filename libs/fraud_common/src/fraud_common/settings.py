"""Typed configuration loaded from environment variables (and an optional .env file).

Every service builds only the settings groups it needs. Each group has its own env
prefix so names never collide, and every value is range-checked at startup: a bad
config fails fast with a clear message instead of misbehaving at 5,000 tx/s.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    frozen=True,
)

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_PLACEHOLDERS = ("change_me", "changeme")


def _no_placeholder(value: SecretStr | None) -> SecretStr | None:
    """Refuse to run with a secret still set to the .env.example placeholder."""
    if value is not None and any(p in value.get_secret_value().lower() for p in _PLACEHOLDERS):
        raise ValueError("still the placeholder from .env.example; set a real secret in .env")
    return value


def _host_list(value: str) -> tuple[str, ...]:
    hosts = tuple(h.strip() for h in value.split(",") if h.strip())
    if not hosts:
        raise ValueError("at least one allowed host is required")
    return hosts


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="LOG_")

    level: LogLevel = "INFO"
    format: Literal["json", "console"] = "json"

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


class KafkaSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="KAFKA_")

    bootstrap_servers: str = Field(default="localhost:29092", min_length=1)
    client_id_prefix: str = Field(default="fraud-stream", pattern=r"^[a-zA-Z0-9._-]{1,64}$")
    admin_timeout_s: float = Field(default=15.0, gt=0, le=120)

    @field_validator("bootstrap_servers")
    @classmethod
    def _valid_hosts(cls, value: str) -> str:
        for entry in value.split(","):
            host, sep, port = entry.strip().rpartition(":")
            if not sep or not host or not port.isdigit() or not 1 <= int(port) <= 65535:
                raise ValueError(f"bootstrap server {entry!r} must look like host:port")
        return value


class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="POSTGRES_")

    host: str = Field(default="localhost", min_length=1)
    port: int = Field(default=5432, ge=1, le=65535)
    db: str = Field(default="fraud", min_length=1)
    user: str = Field(default="fraud", min_length=1)
    password: SecretStr
    connect_timeout_s: int = Field(default=5, ge=1, le=60)
    # Least-privilege login for the dashboard (SELECT/INSERT/UPDATE on alerts only).
    # Created by `python -m fraud_common.admin ensure-db-role` (the db-init service).
    app_user: str = Field(default="fraud_app", pattern=r"^[a-z_][a-z0-9_]{0,62}$")
    app_password: SecretStr | None = Field(default=None, min_length=12)

    _check_password = field_validator("password", "app_password")(_no_placeholder)

    @model_validator(mode="after")
    def _distinct_users(self) -> PostgresSettings:
        if self.app_user == self.user:
            raise ValueError("POSTGRES_APP_USER must differ from POSTGRES_USER")
        return self

    def connect_kwargs(self) -> dict[str, str | int]:
        """Keyword arguments for psycopg.connect; keeps the password out of any DSN string."""
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db,
            "user": self.user,
            "password": self.password.get_secret_value(),
            "connect_timeout": self.connect_timeout_s,
        }

    def app_connect_kwargs(self) -> dict[str, str | int]:
        """Like connect_kwargs, but for the least-privilege application login."""
        if self.app_password is None:
            raise ValueError("POSTGRES_APP_PASSWORD is not set (see .env.example)")
        return self.connect_kwargs() | {
            "user": self.app_user,
            "password": self.app_password.get_secret_value(),
        }


class ProducerSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="PRODUCER_")

    data_dir: str = Field(default="/app/data", min_length=1)
    rate_tps: int = Field(default=1000, ge=1, le=10_000)
    label_delay_s: float = Field(default=10.0, ge=0, le=3600)
    loop: bool = True
    max_events: int = Field(default=0, ge=0)  # 0 = no limit


class ScorerSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="SCORER_")

    data_dir: str = Field(default="/app/data", min_length=1)
    model_api_url: AnyHttpUrl = AnyHttpUrl("http://model-api:8001")
    # Shares the model API's key so there is one secret to manage.
    api_key: SecretStr = Field(
        min_length=32, validation_alias=AliasChoices("SCORER_API_KEY", "MODEL_API_API_KEY")
    )
    group_id: str = Field(default="scorer", pattern=r"^[a-zA-Z0-9._-]{1,64}$")
    metrics_port: int = Field(default=9101, ge=1, le=65535)
    batch_max: int = Field(default=256, ge=1, le=512)
    batch_wait_ms: int = Field(default=5, ge=0, le=1000)
    max_inflight: int = Field(default=2, ge=1, le=8)  # batches scored concurrently
    request_timeout_ms: int = Field(default=2000, ge=10, le=30_000)
    retry_max_backoff_ms: int = Field(default=2000, ge=50, le=60_000)
    # Load each card's training-period history at startup (live features = offline features).
    bootstrap_history: bool = True
    # On restart, re-read already-scored transactions.raw events to rebuild card state.
    restore_state: bool = True
    restore_max_per_partition: int = Field(default=1_000_000, ge=0, le=100_000_000)

    _check_key = field_validator("api_key")(_no_placeholder)


class ModelApiSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="MODEL_API_")

    model_dir: str = Field(default="/app/models", min_length=1)
    api_key: SecretStr = Field(min_length=32)
    max_batch: int = Field(default=512, ge=1, le=512)
    threads: int = Field(default=2, ge=1, le=32)  # keep <= the container's CPU limit
    # Host headers accepted (blocks DNS-rebinding requests from web pages).
    allowed_hosts: str = "localhost,127.0.0.1,model-api"

    _check_key = field_validator("api_key")(_no_placeholder)

    host: str = "0.0.0.0"  # noqa: S104  (inside the container; compose binds the host side to 127.0.0.1)
    port: int = Field(default=8001, ge=1, le=65535)

    @property
    def host_list(self) -> tuple[str, ...]:
        return _host_list(self.allowed_hosts)


class DashboardSettings(BaseSettings):
    model_config = SettingsConfigDict(**_BASE_CONFIG, env_prefix="DASHBOARD_")

    stats_window_s: int = Field(default=60, ge=5, le=3600)
    push_interval_s: float = Field(default=1.0, gt=0, le=60)
    group_id: str = Field(default="dashboard", pattern=r"^[a-zA-Z0-9._-]{1,64}$")
    host: str = "0.0.0.0"  # noqa: S104  (inside the container; compose binds the host side to 127.0.0.1)
    port: int = Field(default=8000, ge=1, le=65535)
    # Host headers accepted (blocks DNS-rebinding requests from web pages).
    allowed_hosts: str = "localhost,127.0.0.1,dashboard"

    @property
    def host_list(self) -> tuple[str, ...]:
        return _host_list(self.allowed_hosts)
