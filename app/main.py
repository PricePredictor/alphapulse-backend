from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.core.logger import logger
from app.core.config import settings
from app.routers import predict  # only import predict if prediction.py is obsolete

# Load environment variables
load_dotenv()

# Define FastAPI app
app = FastAPI(title=settings.PROJECT_NAME)

logger.info("🚀 FastAPI application starting...")

# Apply CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(predict.router)

# For local dev testing only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
