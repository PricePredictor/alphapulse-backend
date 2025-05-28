from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.logger import logger
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

logger.info("🚀 FastAPI application starting...")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
