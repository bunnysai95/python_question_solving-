# settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Auth API"
    # In production, use a strong secret from env/secret manager
    JWT_SECRET: str = "change-me-please-very-secret"
    JWT_ALG: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # For dev with Vite
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    # SQLite for local dev. For Postgres, see the note at the bottom.
    DB_URL: str = "sqlite://db.sqlite3"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
