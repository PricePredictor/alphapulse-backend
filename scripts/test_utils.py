import pandas as pd
import numpy as np

from utils.indicators import calculate_sma, calculate_rsi
from utils.preprocessing import normalize_dataframe

data = pd.Series([100 + i for i in range(20)])

print("=== Testing SMA ===")
print(calculate_sma(data, window=5))

print("=== Testing RSI ===")
print(calculate_rsi(data, window=5))

print("=== Testing Normalization ===")
df = pd.DataFrame({
    "Close": np.linspace(100, 110, 10),
    "Volume": np.random.randint(1000000, 5000000, 10)
})
print(normalize_dataframe(df))

print("=== Testing Empty DataFrame ===")
empty_df = pd.DataFrame()
print(normalize_dataframe(empty_df))
