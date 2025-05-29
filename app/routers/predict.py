from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import timedelta
import pandas as pd
import joblib
import numpy as np

from app.utils.features import fetch_hourly_data

router = APIRouter()

class PredictRequest(BaseModel):
    ticker: str
    horizon_hours: int = 24  # default: 1 trading day

@router.post("/predict")
def predict_price(req: PredictRequest):
    try:
        df = fetch_hourly_data(req.ticker, hours_back=100)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data fetch error: {str(e)}")

    try:
        model = joblib.load("models/xgb_model.pkl")
    except:
        raise HTTPException(status_code=500, detail="Model not found or failed to load.")

    forecast = []
    current_features = df.iloc[-1:].copy()
    current_time = df.index[-1]

    for i in range(req.horizon_hours):
        X = current_features.drop(columns=["close"])  # assuming 'close' is the label
        y_pred = model.predict(X)[0]

        current_time += timedelta(hours=1)
        forecast.append({
            "datetime": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "price": round(float(y_pred), 2)
        })

        # Create next input row based on last + prediction
        new_row = current_features.copy()
        new_row["lag_3"] = new_row["lag_2"]
        new_row["lag_2"] = new_row["lag_1"]
        new_row["lag_1"] = y_pred
        new_row["close"] = y_pred
        new_row["hour"] = current_time.hour
        new_row["dayofweek"] = current_time.weekday()
        new_row.index = [current_time]
        current_features = new_row

    return {
        "ticker": req.ticker.upper(),
        "predictions": forecast
    }
