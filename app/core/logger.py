from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO", colorize=True)
logger.add("logs/runtime.log", rotation="1 MB", retention="7 days", level="INFO")
