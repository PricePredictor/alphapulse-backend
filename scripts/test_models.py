import pandas as pd
import numpy as np
from models.xgboost_model import predict as xgb_predict
from models.lstm_model import predict as lstm_predict
from models.linear_model import predict as linear_predict

# XGBoost input
xgb_data = pd.DataFrame({
    "SMA_10 ": np.linspace(100, 110, 10),
    "SMA_50 ": np.linspace(100, 105, 10),
    "RSI ": np.random.uniform(30, 70, 10),
})

# LSTM and Linear input
lstm_data = pd.DataFrame({
    "Close": np.linspace(100, 110, 50),
    "Volume": np.random.randint(1000000, 5000000, 50),
    "SMA_5": np.linspace(100, 109, 50),
    "RSI": np.random.uniform(30, 70, 50),
})

print("=== Testing XGBoost ===")
print("[Step 2] Calling XGBoost model")
print(xgb_predict("AAPL", xgb_data))

print("=== Testing LSTM ===")
print("[Step 2] Calling LSTM model")
try:
    print(lstm_predict("AAPL", lstm_data))
except Exception as e:
    print("LSTM failed:", e)

print("=== Testing Linear Model ===")
print("[Step 2] Calling Linear model")
print(linear_predict("AAPL", lstm_data))
