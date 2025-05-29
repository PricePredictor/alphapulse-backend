import pandas as pd
import numpy as np
from models.xgboost_model import predict as xgb_predict
from models.lstm_model import predict as lstm_predict
from models.linear_model import predict as linear_predict

# Create dummy input data
dummy_data = pd.DataFrame({
    "Close": np.linspace(100, 110, 10),
    "Volume": np.random.randint(1000000, 5000000, 10),
    "SMA_5": np.linspace(100, 109, 10),
    "RSI": np.random.uniform(30, 70, 10),
})

print("=== Testing XGBoost ===")
print(xgb_predict("AAPL", dummy_data))

print("=== Testing LSTM ===")
print(lstm_predict("AAPL", dummy_data))

print("=== Testing Linear Model ===")
print(linear_predict("AAPL", dummy_data))
