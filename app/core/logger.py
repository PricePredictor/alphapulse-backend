import os
from loguru import logger

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", colorize=True, format="{time} {level} {message}")
logger.add("logs/runtime.log", rotation="1 MB", retention="7 days", level="INFO")
