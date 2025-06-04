import pandas as pd
import numpy as np
import ta

def calculate_sma(series: pd.Series, window: int = 14) -> pd.Series:
    try:
        return series.rolling(window=window).mean()
    except Exception as e:
        print(f"[SMA Error] {e}")
        return pd.Series([np.nan] * len(series))

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    try:
        return ta.momentum.RSIIndicator(close=series, window=window).rsi()
    except Exception as e:
        print(f"[RSI Error] {e}")
        return pd.Series([np.nan] * len(series))
