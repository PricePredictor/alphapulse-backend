
import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta

def fetch_hourly_data(ticker: str, hours_back: int = 100):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours_back * 2)  # buffer for indicator lookback

    df = yf.download(tickers=ticker, interval="60m", start=start_time.strftime('%Y-%m-%d'), progress=False)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", 
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
    })

    df["sma_5"] = ta.trend.sma_indicator(df["close"], window=5)
    df["sma_10"] = ta.trend.sma_indicator(df["close"], window=10)
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    df["macd"] = ta.trend.macd(df["close"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek

    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)
    df["lag_3"] = df["close"].shift(3)

    df["volume_change"] = df["volume"].pct_change()

    df.dropna(inplace=True)
    return df
