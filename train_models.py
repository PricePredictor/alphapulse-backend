# ======================
# train_models.py
# ======================
# This script trains XGBoost, LSTM, Random Forest, and LightGBM models
# and saves them to disk for production use.

import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from ta.trend import sma_indicator
from ta.momentum import rsi
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# ---------- Train XGBoost Model ----------
def train_xgboost_model(ticker="AAPL"):
    df = yf.download(ticker, period="1y", interval="1d")
    df.dropna(inplace=True)
    df['SMA_10'] = sma_indicator(close=df['Close'], window=10)
    df['SMA_50'] = sma_indicator(close=df['Close'], window=50)
    df['RSI'] = rsi(close=df['Close'], window=14)
    df.dropna(inplace=True)

    X = df[['SMA_10', 'SMA_50', 'RSI']]
    y = df['Close'].shift(-1)
    X = X[:-1]
    y = y[:-1]

    model = XGBRegressor(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, "xgb_model.pkl")

# ---------- Train LSTM Model ----------
def train_lstm_model(ticker="AAPL", sequence_length=50):
    df = yf.download(ticker, period="1y", interval="1d")
    close_prices = df[['Close']].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10
