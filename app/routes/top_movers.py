# app/routes/top_movers.py

from fastapi import APIRouter
from scripts.top_movers import get_daily_pct_change, TICKERS

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
