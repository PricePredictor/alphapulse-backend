from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AlphaPulse AI"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_ROTATION: str = "1 MB"
    LOG_RETENTION: str = "7 days"

    class Config:
        env_file = ".env"

settings = Settings()
