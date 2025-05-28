from pydantic import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str
    LOG_LEVEL: str
    LOG_ROTATION: str
    LOG_RETENTION: str

    class Config:
        env_file = ".env"

settings = Settings()
