from dotenv import load_dotenv
import os

load_dotenv()  # Load values from .env

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "AlphaPulse")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    API_VERSION: str = os.getenv("API_VERSION", "v1")

settings = Settings()
