import sys
import os
from loguru import logger

# Ensure logs directory exists
LOG_DIR = "logs"
LOG_FILE = f"{LOG_DIR}/runtime.log"
os.makedirs(LOG_DIR, exist_ok=True)

# Clear default logger and configure outputs
logger.remove()
logger.add(sys.stdout, level="INFO", colorize=True, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
logger.add(LOG_FILE, rotation="1 MB", retention="7 days", level="INFO")
