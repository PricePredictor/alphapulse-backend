from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import mean_squared_error
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Initialize FastAPI
app = FastAPI()

# Allow CORS (relax in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    xgb_model: XGBRegressor = joblib.load("xgb_model.pkl")
    lstm_model = load_model("lstm_model.h5")
    lstm_scaler: MinMaxScaler = joblib.load("lstm_scaler.save")
    rf_model = joblib.load("random_forest.pkl")
    lgb_model = joblib.load("lightgbm.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# ============================
# Endpoint: Predict
# ============================
@app.get("/predict")
def predict_price(ticker: str, model_type: str = "xgb"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")

        df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df.dropna(inplace=True)

        X = df[['SMA_10', 'SMA_50', 'RSI']].copy()
        X.columns = ['SMA10', 'SMA50', 'RSI']
        X.columns = X.columns.str.strip()
        last_row = X.iloc[-1:].values

        if model_type == "xgb":
            prediction = xgb_model.predict(last_row)[0]
        elif model_type == "rf":
            prediction = rf_model.predict(last_row)[0]
        elif model_type == "lgb":
            prediction = lgb_model.predict(last_row)[0]
        elif model_type == "lstm":
            scaled = lstm_scaler.transform(df[["Close"]].values)
            last_sequence = scaled[-50:].reshape(1, 50, 1)
            prediction = lstm_scaler.inverse_transform(lstm_model.predict(last_sequence))[0][0]
        else:
            raise HTTPException(status_code=400, detail="Invalid or missing model type")

        return {
            "ticker": ticker.upper(),
            "model": model_type.upper(),
            "predicted_price": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Endpoint: Backtest All Models
# ============================
@app.get("/backtest-multi")
def backtest_multi(ticker: str = "AAPL", start: str = "2023-01-01", end: str = "2023-03-01"):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d")
        df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df.dropna(inplace=True)

        X = df[['SMA_10', 'SMA_50', 'RSI']].copy()
        X.columns = ['SMA10', 'SMA50', 'RSI']
        X.columns = X.columns.str.strip()
        y_true = df['Close'].values.ravel()
        results = {}

        preds_xgb = xgb_model.predict(X).ravel()
        results["XGBoost"] = {"mse": round(mean_squared_error(y_true, preds_xgb), 4)}

        preds_rf = rf_model.predict(X).ravel()
        results["RandomForest"] = {"mse": round(mean_squared_error(y_true, preds_rf), 4)}

        preds_lgb = lgb_model.predict(X).ravel()
        results["LightGBM"] = {"mse": round(mean_squared_error(y_true, preds_lgb), 4)}

        scaled_close = lstm_scaler.transform(df[["Close"]].values)
        sequence_length = 50
        preds_lstm = []
        actual_lstm = []

        for i in range(sequence_length, len(scaled_close)):
            X_seq = scaled_close[i-sequence_length:i].reshape(1, sequence_length, 1)
            pred_scaled = lstm_model.predict(X_seq)
            pred = lstm_scaler.inverse_transform(pred_scaled)[0][0]
            preds_lstm.append(pred)
            actual_lstm.append(df["Close"].values[i])

        results["LSTM"] = {
            "mse": round(mean_squared_error(np.array(actual_lstm).ravel(), np.array(preds_lstm).ravel()), 4)
        }

        return {
            "ticker": ticker,
            "start": start,
            "end": end,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Endpoint: Predict Ensemble
# ============================
@app.get("/predict-ensemble")
def predict_ensemble(ticker: str = "AAPL"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")

        df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df.dropna(inplace=True)

        features = df[['SMA_10', 'SMA_50', 'RSI']].copy()
        features.columns = ['SMA10', 'SMA50', 'RSI']
        features.columns = features.columns.str.strip()
        last_row = features.iloc[-1:].values
        close_prices = df['Close'].values

        predictions = {}
        mse_scores = {}

        preds_xgb = xgb_model.predict(features).ravel()
        mse_scores['XGBoost'] = mean_squared_error(close_prices[-len(preds_xgb):], preds_xgb)
        predictions['XGBoost'] = float(xgb_model.predict(last_row)[0])

        preds_rf = rf_model.predict(features).ravel()
        mse_scores['RandomForest'] = mean_squared_error(close_prices[-len(preds_rf):], preds_rf)
        predictions['RandomForest'] = float(rf_model.predict(last_row)[0])

        preds_lgb = lgb_model.predict(features).ravel()
        mse_scores['LightGBM'] = mean_squared_error(close_prices[-len(preds_lgb):], preds_lgb)
        predictions['LightGBM'] = float(lgb_model.predict(last_row)[0])

        scaled_close = lstm_scaler.transform(df[['Close']].values)
        sequence_length = 50
        X_lstm = []
        y_lstm = []
        for i in range(sequence_length, len(scaled_close)):
            X_lstm.append(scaled_close[i-sequence_length:i])
            y_lstm.append(close_prices[i])
        X_lstm = np.array(X_lstm).reshape(-1, sequence_length, 1)
        y_lstm = np.array(y_lstm).ravel()

        preds_lstm = lstm_model.predict(X_lstm)
        preds_lstm = lstm_scaler.inverse_transform(preds_lstm).ravel()
        mse_scores['LSTM'] = mean_squared_error(y_lstm, preds_lstm)

        last_seq = scaled_close[-sequence_length:].reshape(1, sequence_length, 1)
        predictions['LSTM'] = float(lstm_scaler.inverse_transform(lstm_model.predict(last_seq))[0][0])

        ensemble_avg = round(np.mean(list(predictions.values())), 2)
        inv_mses = {k: 1 / v for k, v in mse_scores.items()}
        total_inv = sum(inv_mses.values())
        weights = {k: inv_mses[k] / total_inv for k in inv_mses}
        ensemble_weighted = round(sum(predictions[k] * weights[k] for k in predictions), 2)

        return {
            "ticker": ticker.upper(),
            "predictions": {k: round(v, 2) for k, v in predictions.items()},
            "weights": {k: round(weights[k], 4) for k in weights},
            "mse_scores": {k: round(mse_scores[k], 4) for k in mse_scores},
            "ensemble_avg": ensemble_avg,
            "ensemble_weighted": ensemble_weighted
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Endpoint: Accuracy Multi
# ============================
@app.get("/accuracy-multi")
def accuracy_multi(ticker: str = "AAPL"):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df.dropna(inplace=True)

        X = df[['SMA_10', 'SMA_50', 'RSI']].copy()
        X.columns = ['SMA10', 'SMA50', 'RSI']
        X.columns = X.columns.str.strip()
        y_true = df['Close'].values.ravel()

        results = {}
        results["XGBoost"] = round(mean_squared_error(y_true, xgb_model.predict(X).ravel()), 4)
        results["RandomForest"] = round(mean_squared_error(y_true, rf_model.predict(X).ravel()), 4)
        results["LightGBM"] = round(mean_squared_error(y_true, lgb_model.predict(X).ravel()), 4)

        scaled_close = lstm_scaler.transform(df[['Close']].values)
        sequence_length = 50
        preds_lstm = []
        actual_lstm = []

        for i in range(sequence_length, len(scaled_close)):
            X_seq = scaled_close[i-sequence_length:i].reshape(1, sequence_length, 1)
            pred_scaled = lstm_model.predict(X_seq)
            pred = lstm_scaler.inverse_transform(pred_scaled)[0][0]
            preds_lstm.append(pred)
            actual_lstm.append(df['Close'].values[i])

        results["LSTM"] = round(mean_squared_error(np.array(actual_lstm).ravel(), np.array(preds_lstm).ravel()), 4)

        return {
            "ticker": ticker.upper(),
            "accuracy": results,
            "n_samples": len(y_true)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
