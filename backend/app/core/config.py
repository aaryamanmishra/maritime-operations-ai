
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    PROJECT_NAME: str = "Maritime Operations AI"

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:80",
        "http://localhost",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        return v

    # Database (PostgreSQL + PostGIS)
    DATABASE_URL: str = "postgresql+asyncpg://maritime_user:maritime_secure_dev_password@postgres:5432/maritime_ops"

    # Redis (Streams, Pub/Sub, Primary Cache)
    REDIS_URL: str = "redis://redis:6379/0"

    # AISStream External Feed
    AISSTREAM_API_KEY: str = ""
    AISSTREAM_URL: str = "wss://stream.aisstream.io/v0/stream"

    # Initial AIS Bounding Box (English Channel / Dover Strait default)
    AIS_BBOX_MIN_LAT: float = 49.5
    AIS_BBOX_MIN_LON: float = -2.0
    AIS_BBOX_MAX_LAT: float = 51.5
    AIS_BBOX_MAX_LON: float = 2.5

    # Redis Stream & Channel Keys
    AIS_STREAM_KEY: str = "stream:ais:raw"
    AIS_CONSUMER_GROUP: str = "group:ais:persistence"
    AIS_CONSUMER_NAME: str = "consumer:persist_worker_1"
    AIS_BROADCAST_CHANNEL: str = "channel:ais:broadcast"
    AIS_STATUS_REDIS_KEY: str = "ais:pipeline:metrics"

    # Position History Thinning / Deadband Thresholds
    AIS_HISTORY_DISTANCE_THRESHOLD_M: float = 100.0
    AIS_HISTORY_HEADING_THRESHOLD_DEG: float = 5.0
    AIS_HISTORY_TIME_THRESHOLD_S: int = 180

    # Copernicus CDSE Configuration
    CDSE_CLIENT_ID: str = ""
    CDSE_CLIENT_SECRET: str = ""
    CDSE_USERNAME: str = ""
    CDSE_PASSWORD: str = ""
    CDSE_STAC_URL: str = "https://stac.dataspace.copernicus.eu/v1/"
    CDSE_TOKEN_URL: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    CDSE_ODATA_URL: str = "https://catalogue.dataspace.copernicus.eu/odata/v1"

    # SAR Model & Correlation
    SAR_MODEL_WEIGHTS_PATH: str = "ml/weights/yolo26s_sar_vessel.pt"
    SAR_CORRELATION_TIME_WINDOW_MINUTES: int = 15
    SAR_CORRELATION_DISTANCE_KM: float = 3.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
