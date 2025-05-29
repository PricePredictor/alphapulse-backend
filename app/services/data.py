# services/data.py

import yfinance as yf

def get_clean_data(ticker: str, period: str = "60d"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            return None
        return df
    except Exception:
        return None
