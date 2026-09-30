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
