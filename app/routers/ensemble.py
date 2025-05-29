from fastapi import APIRouter, HTTPException, Query
import traceback

from app.models.xgb_model import predict_xgb
from app.models.lstm_model import predict_lstm
from app.models.linear_model import predict_linear

router = APIRouter()

@router.get("/predict-ensemble")
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
