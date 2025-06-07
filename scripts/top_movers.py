import yfinance as yf
import pandas as pd
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD"]
def get_daily_pct_change(ticker_list):
    data = {}
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period="5d", interval="1d")  # enough to get last 2 real days
            df = df.tail(2)  # Last two rows
            if len(df) == 2:
                change = ((df["Close"].iloc[1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
                data[ticker] = round(change, 2)
        except Exception as e:
            print(f"[{ticker}] Error fetching data: {e}")
    return data

from fastapi import APIRouter
from top_movers import get_daily_pct_change, TICKERS

router = APIRouter()

@router.get("/top-movers")
def top_movers():
    changes = get_daily_pct_change(TICKERS)
    sorted_tickers = sorted(changes.items(), key=lambda x: x[1], reverse=True)

    gainers = sorted_tickers[:5]
    losers = sorted_tickers[-5:][::-1]

    return {
        "gainers": [{"ticker": g[0], "change": g[1]} for g in gainers],
        "losers": [{"ticker": l[0], "change": l[1]} for l in losers]
    }
