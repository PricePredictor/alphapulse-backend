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

# Predict Enesamble
from fastapi import FastAPI, HTTPException, Query
from typing import Dict
import traceback

app = FastAPI()

@app.get("/predict-ensemble")
def predict_ensemble(ticker: str = Query(..., description="Ticker symbol (e.g., AAPL)")):
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    results = {}
    
    try:
        results['xgb'] = predict_xgb(ticker)
    except Exception as e:
        print("XGB failed:", e)
        traceback.print_exc()

    try:
        results['lstm'] = predict_lstm(ticker)
    except Exception as e:
        print("LSTM failed:", e)
        traceback.print_exc()

    try:
        results['linear'] = predict_linear(ticker)
    except Exception as e:
        print("Linear failed:", e)
        traceback.print_exc()

    if not results:
        raise HTTPException(status_code=500, detail="All models failed")

    ensemble = sum(results.values()) / len(results)

    return {
        "ticker": ticker.upper(),
        "ensemble": round(ensemble, 2),
        "models": {k: round(v, 2) for k, v in results.items()}
    }

