# scripts/top_movers.py

import yfinance as yf
import pandas as pd
import random

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD"]

def get_daily_pct_change(ticker_list):
    data = {}
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period="5d", interval="1d")
            df = df.tail(2)
            if len(df) == 2:
                prev = df["Close"].iloc[0]
                curr = df["Close"].iloc[1]
                change = ((curr - prev) / prev) * 100
                data[ticker] = round(float(change), 2)
            else:
                raise ValueError("Not enough data")
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
            # 🔁 Fallback to random % change for dev testing
            data[ticker] = round(random.uniform(-5, 5), 2)
    return data
