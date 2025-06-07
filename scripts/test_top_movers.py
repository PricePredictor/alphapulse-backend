from top_movers import get_daily_pct_change, TICKERS
import json

result = get_daily_pct_change(TICKERS)
sorted_tickers = sorted(result.items(), key=lambda x: x[1], reverse=True)
print("Top Gainers:", json.dumps(sorted_tickers[:5], indent=2))
print("Top Losers:", json.dumps(sorted_tickers[-5:][::-1], indent=2))
