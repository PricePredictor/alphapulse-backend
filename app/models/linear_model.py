import numpy as np

def predict(ticker: str, data):
    try:
        prices = data["Close"].values
        if len(prices) < 2:
            raise ValueError("Not enough data")
        trend = prices[-1] - prices[-2]
        return float(prices[-1] + trend)
    except Exception as e:
        print(f"[Linear] Error: {e}")
        return np.nan
