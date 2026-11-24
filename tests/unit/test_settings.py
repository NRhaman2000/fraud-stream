from pathlib import Path

import pytest
from pydantic import ValidationError

from fraud_common.settings import (
    DashboardSettings,
    KafkaSettings,
    LoggingSettings,
    ModelApiSettings,
    PostgresSettings,
    ProducerSettings,
    ScorerSettings,
)


class TestLogging:
    def test_defaults(self) -> None:
        s = LoggingSettings()
        assert (s.level, s.format) == ("INFO", "json")

    def test_level_is_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        assert LoggingSettings().level == "DEBUG"

    @pytest.mark.parametrize(("key", "value"), [("LOG_LEVEL", "LOUD"), ("LOG_FORMAT", "xml")])
    def test_rejects_unknown_values(
        self, monkeypatch: pytest.MonkeyPatch, key: str, value: str
    ) -> None:
        monkeypatch.setenv(key, value)
        with pytest.raises(ValidationError):
            LoggingSettings()

    def test_settings_are_immutable(self) -> None:
        s = LoggingSettings()
        with pytest.raises(ValidationError):
            s.level = "ERROR"


class TestKafka:
    @pytest.mark.parametrize("servers", ["kafka:9092", "a:1,b:65535", "10.0.0.1:29092"])
    def test_accepts_host_port_lists(self, monkeypatch: pytest.MonkeyPatch, servers: str) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", servers)
        assert KafkaSettings().bootstrap_servers == servers

    @pytest.mark.parametrize(
        "servers", ["kafka", "kafka:", ":9092", "kafka:0", "kafka:65536", "kafka:abc", "a:1,,b:2"]
    )
    def test_rejects_bad_servers(self, monkeypatch: pytest.MonkeyPatch, servers: str) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", servers)
        with pytest.raises(ValidationError):
            KafkaSettings()

    def test_rejects_unsafe_client_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_CLIENT_ID_PREFIX", "bad id;rm")
        with pytest.raises(ValidationError):
            KafkaSettings()


class TestPostgres:
    def test_password_is_required(self) -> None:
        with pytest.raises(ValidationError, match="password"):
            PostgresSettings()

    def test_password_never_appears_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret-value")
        s = PostgresSettings()
        assert "s3cret-value" not in repr(s)
        assert "s3cret-value" not in str(s.model_dump())

    def test_connect_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
        monkeypatch.setenv("POSTGRES_HOST", "db")
        assert PostgresSettings().connect_kwargs() == {
            "host": "db",
            "port": 5432,
            "dbname": "fraud",
            "user": "fraud",
            "password": "pw",
            "connect_timeout": 5,
        }

    @pytest.mark.parametrize("port", ["0", "65536", "-1", "x"])
    def test_rejects_bad_port(self, monkeypatch: pytest.MonkeyPatch, port: str) -> None:
        monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
        monkeypatch.setenv("POSTGRES_PORT", port)
        with pytest.raises(ValidationError):
            PostgresSettings()

    def test_reads_dotenv_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        Path(".env").write_text("POSTGRES_PASSWORD=from-file\n", encoding="utf-8")
        assert PostgresSettings().password.get_secret_value() == "from-file"


class TestServiceRanges:
    @pytest.mark.parametrize(
        ("cls", "env", "ok", "bad"),
        [
            (ProducerSettings, "PRODUCER_RATE_TPS", ["1", "10000"], ["0", "10001"]),
            (ProducerSettings, "PRODUCER_LABEL_DELAY_S", ["0", "3600"], ["-0.1", "3601"]),
            (ScorerSettings, "SCORER_BATCH_MAX", ["1", "512"], ["0", "513"]),
            (ScorerSettings, "SCORER_BATCH_WAIT_MS", ["0", "1000"], ["-1", "1001"]),
            (DashboardSettings, "DASHBOARD_STATS_WINDOW_S", ["5", "3600"], ["4", "3601"]),
            (DashboardSettings, "DASHBOARD_PUSH_INTERVAL_S", ["0.1", "60"], ["0", "61"]),
        ],
    )
    def test_boundaries(
        self, monkeypatch: pytest.MonkeyPatch, cls: type, env: str, ok: list[str], bad: list[str]
    ) -> None:
        monkeypatch.setenv("MODEL_API_API_KEY", "x" * 32)
        for value in ok:
            monkeypatch.setenv(env, value)
            cls()
        for value in bad:
            monkeypatch.setenv(env, value)
            with pytest.raises(ValidationError):
                cls()

    def test_scorer_url_must_be_http(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_API_API_KEY", "x" * 32)
        monkeypatch.setenv("SCORER_MODEL_API_URL", "ftp://model-api")
        with pytest.raises(ValidationError):
            ScorerSettings()

    def test_model_api_key_minimum_length(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_API_API_KEY", "x" * 31)
        with pytest.raises(ValidationError):
            ModelApiSettings()
        monkeypatch.setenv("MODEL_API_API_KEY", "x" * 32)
        assert ModelApiSettings().api_key.get_secret_value() == "x" * 32

    def test_scorer_reads_shared_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError):
            ScorerSettings()
        monkeypatch.setenv("MODEL_API_API_KEY", "y" * 32)
        assert ScorerSettings().api_key.get_secret_value() == "y" * 32
        monkeypatch.setenv("SCORER_API_KEY", "z" * 32)
        assert ScorerSettings().api_key.get_secret_value() == "z" * 32

    def test_model_api_key_required(self) -> None:
        with pytest.raises(ValidationError):
            ModelApiSettings()


class TestSecrets:
    def test_placeholder_secrets_are_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POSTGRES_PASSWORD", "CHANGE_ME_postgres_password")
        with pytest.raises(ValidationError, match="placeholder"):
            PostgresSettings()
        monkeypatch.setenv("POSTGRES_PASSWORD", "a-real-password")
        monkeypatch.setenv("POSTGRES_APP_PASSWORD", "changeme-app-password")
        with pytest.raises(ValidationError, match="placeholder"):
            PostgresSettings()
        monkeypatch.setenv("MODEL_API_API_KEY", "CHANGE_ME_generate_a_random_key_of_32_chars")
        with pytest.raises(ValidationError, match="placeholder"):
            ModelApiSettings()
        with pytest.raises(ValidationError, match="placeholder"):
            ScorerSettings()

    def test_app_login(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POSTGRES_PASSWORD", "admin-password")
        with pytest.raises(ValueError, match="POSTGRES_APP_PASSWORD"):
            PostgresSettings().app_connect_kwargs()
        monkeypatch.setenv("POSTGRES_APP_PASSWORD", "app-password-123")
        kw = PostgresSettings().app_connect_kwargs()
        assert (kw["user"], kw["password"]) == ("fraud_app", "app-password-123")
        assert PostgresSettings().connect_kwargs()["user"] == "fraud"
        monkeypatch.setenv("POSTGRES_APP_USER", "fraud")
        with pytest.raises(ValidationError, match="must differ"):
            PostgresSettings()
        monkeypatch.setenv("POSTGRES_APP_USER", "Robert'); DROP")
        with pytest.raises(ValidationError):
            PostgresSettings()

    def test_allowed_hosts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DASHBOARD_ALLOWED_HOSTS", " a.local , b ")
        assert DashboardSettings().host_list == ("a.local", "b")
        monkeypatch.setenv("DASHBOARD_ALLOWED_HOSTS", " , ")
        with pytest.raises(ValueError, match="allowed host"):
            _ = DashboardSettings().host_list
