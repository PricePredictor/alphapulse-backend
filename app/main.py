from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.core.logger import logger
from app.core.config import settings
from app.routers import fastAPI_endpoint  # Update this line based on your actual router filename

# Load environment variables
load_dotenv()

app = FastAPI(title=settings.PROJECT_NAME)

logger.info("🚀 FastAPI application starting...")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your router
app.include_router(fastAPI_endpoint.router)
