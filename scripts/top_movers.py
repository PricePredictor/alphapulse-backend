# scripts/top_movers.py

import yfinance as yf

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD"]

def get_daily_pct_change(ticker_list):
    data = {}
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period="5d", interval="1d", progress=False)
            df = df.tail(2)
            if len(df) == 2:
                prev_close = df["Close"].iloc[0]
                last_close = df["Close"].iloc[1]
                pct_change = ((last_close - prev_close) / prev_close) * 100
                data[ticker] = round(pct_change, 2)
        except Exception as e:
            print(f"[{ticker}] Error fetching data: {e}")
    return data
