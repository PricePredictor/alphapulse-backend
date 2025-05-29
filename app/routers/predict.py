from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List

from app.utils.features import fetch_hourly_data

router = APIRouter()

class PredictRequest(BaseModel):
    ticker: str
    horizon_hours: int = 24  # default: 1 trading day

class PredictionItem(BaseModel):
    datetime: datetime
    price: float

class PredictResponse(BaseModel):
    ticker: str
    predictions: List[PredictionItem]


@router.post("/predict", response_model=PredictResponse)
def predict_price(req: PredictRequest):
    try:
        df = fetch_hourly_data(req.ticker, hours_back=100)
        # Mocked model output — replace with real model
        future_times = df.index[-req.horizon_hours:]
        preds = df["Close"].tail(req.horizon_hours).values.tolist()

        results = [
            {"datetime": str(ts), "price": float(price)}
            for ts, price in zip(future_times, preds)
        ]
        return {
            "ticker": req.ticker,
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
