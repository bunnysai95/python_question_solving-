from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Auth API"
    #  In production, use a stong password in prod env 
    JWT_SECRET: str = "change-me in production " 
    JWT_ALG: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 
    # for dev with vite
    CORS_ORGINS: list[str] = ["http://localhost:5173"]
    # sqlite for local dev
    DB_URL:str = "sqlite://db.sqlite3"
    model_config = SettingsConfigDict(env_file = ".env", extra = 'ignore')
settings = Settings()