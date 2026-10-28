from app.core.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    settings = Settings()
    assert settings.ENVIRONMENT == "development"
    assert settings.APP_PORT == 8000
    assert "http://localhost:5173" in settings.CORS_ORIGINS
    assert "postgresql+asyncpg" in settings.DATABASE_URL
    assert "redis://" in settings.REDIS_URL


def test_settings_cors_string_parsing():
    settings = Settings(CORS_ORIGINS="http://example.com, http://test.com")
    assert "http://example.com" in settings.CORS_ORIGINS
    assert "http://test.com" in settings.CORS_ORIGINS


def test_credential_stripping_and_sanitization():
    settings = Settings(
        CDSE_USERNAME="  user@example.com  ",
        CDSE_PASSWORD="'secret_with_single_quotes'",
        CDSE_CLIENT_ID=' "client-id-123" ',
        CDSE_CLIENT_SECRET="secret_token",
        AISSTREAM_API_KEY="  'api_key_456'  ",
    )
    assert settings.CDSE_USERNAME == "user@example.com"
    assert settings.CDSE_PASSWORD == "secret_with_single_quotes"
    assert settings.CDSE_CLIENT_ID == "client-id-123"
    assert settings.CDSE_CLIENT_SECRET == "secret_token"
    assert settings.AISSTREAM_API_KEY == "api_key_456"


def test_credential_none_handling():
    settings = Settings(
        CDSE_USERNAME=None,
        CDSE_PASSWORD=None,
        CDSE_CLIENT_ID=None,
        CDSE_CLIENT_SECRET=None,
        AISSTREAM_API_KEY=None,
    )
    assert settings.CDSE_USERNAME == ""
    assert settings.CDSE_PASSWORD == ""
    assert settings.CDSE_CLIENT_ID == ""
    assert settings.CDSE_CLIENT_SECRET == ""
    assert settings.AISSTREAM_API_KEY == ""
