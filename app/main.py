from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.core.logger import logger
from app.core.config import settings
from app.routers import prediction  # ✅ updated from fastAPI_endpoint

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
app.include_router(prediction.router)  # ✅ updated from fastAPI_endpoint


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
