# scripts/top_movers.py

import yfinance as yf

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD"]

def get_daily_pct_change(ticker_list):
    data = {}
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period="5d", interval="1d")
            df = df.tail(2)
            print(f"{ticker} close values:\n{df['Close']}")

            if len(df) == 2:
                prev = df["Close"].iloc[0]
                curr = df["Close"].iloc[1]
                print(f"{ticker} prev: {prev}, curr: {curr}")
                change = ((curr - prev) / prev) * 100
                data[ticker] = round(float(change), 2)  # 💥 Explicit float here
        except Exception as e:
            print(f"[{ticker}] Error fetching data: {e}")
    return data
