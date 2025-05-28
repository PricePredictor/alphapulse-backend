import sys
import os
from loguru import logger
from app.core.config import settings

# Ensure logs directory exists
LOG_DIR = "logs"
LOG_FILE = f"{LOG_DIR}/runtime.log"
os.makedirs(LOG_DIR, exist_ok=True)

# Clear default loggers and configure custom ones
logger.remove()
logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,
    colorize=True,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
logger.add(
    LOG_FILE,
    rotation=settings.LOG_ROTATION,
    retention=settings.LOG_RETENTION,
    level=settings.LOG_LEVEL
)
