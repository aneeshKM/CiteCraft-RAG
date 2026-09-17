import pytest

from citecraft.config import Settings


def test_openai_response_storage_defaults_to_enabled(monkeypatch):
    monkeypatch.setattr("citecraft.config.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_STORE_RESPONSES", raising=False)

    settings = Settings.from_env(api_key="test-key")

    assert settings.openai_store_responses is True


def test_openai_response_storage_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OPENAI_STORE_RESPONSES", "false")

    settings = Settings.from_env(api_key="test-key")

    assert settings.openai_store_responses is False


def test_openai_response_storage_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("OPENAI_STORE_RESPONSES", "sometimes")

    with pytest.raises(ValueError, match="OPENAI_STORE_RESPONSES must be one of"):
        Settings.from_env(api_key="test-key")
