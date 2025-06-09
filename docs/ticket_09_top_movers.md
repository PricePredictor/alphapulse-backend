# ✅ Ticket #9 – /top-movers Endpoint: Top Gainers & Losers

## 🧾 Description
Implement a FastAPI endpoint `/top-movers` that returns the top 5 gainers and top 5 losers from a predefined stock ticker list based on daily % price change using yfinance data.

---

## 📦 Deliverables

- [x] `TICKERS` list hardcoded inside `scripts/top_movers.py`  
- [x] Logic to download last two days of stock prices using `yfinance`
- [x] Function to compute % daily change and return sorted dictionary
- [x] `/top-movers` FastAPI endpoint in `app/routers/top_movers.py`
- [x] JSON response with two lists: `gainers` and `losers`
- [x] Test script `scripts/test_top_movers.py` to validate local behavior

---

## 🧪 Acceptance Criteria & Tests

| Test Scenario      | How to Test                          | Expected Result                          |
|--------------------|--------------------------------------|-------------------------------------------|
| Endpoint call      | GET `/top-movers` via FastAPI        | Returns JSON with `gainers`, `losers`     |
| Network access     | Use yfinance to download tickers     | Returns data or error-handled fallback    |
| Output structure   | View returned JSON                   | Valid JSON with two keys: gainers, losers |
| Script validation  | `python scripts/test_top_movers.py`  | Prints sorted output of 5 gainers/losers  |

---

## 📂 File Changes

### `scripts/top_movers.py`
- `TICKERS`: hardcoded list of 10 major tech tickers.
- `get_daily_pct_change()`: logic to calculate % change between last 2 close prices.
- Added fallback value of `0.0` in case of Yahoo block or empty dataframe for VM compatibility.

### `app/routers/top_movers.py`
- New router `/top-movers` using FastAPI.
- Fetches from `get_daily_pct_change()`, sorts, and returns JSON with top 5 gainers and losers.

### `scripts/test_top_movers.py`
- CLI script to validate sorted results outside FastAPI.

---

## 🧱 Final Folder Structure

```
alphapulse-backend/
├── app/
│   └── routers/
│       └── top_movers.py
├── scripts/
│   ├── top_movers.py
│   └── test_top_movers.py
```

---

## 🧩 Challenges Faced

| Challenge | Details | Resolution |
|----------|---------|------------|
| 🔒 SSH Key Missing | PEM file path was invalid post new folder clone | Moved file to correct folder, used updated path |
| 📦 yfinance blocked on VM | `yfinance.download()` failed due to Yahoo finance blocking cloud IPs | Added `try-except` fallback with hardcoded dummy prices for VM debug |
| ❌ Git errors | "not a git repo", untracked changes on pull | Configured Git author globally, staged/committed locally first, restored file before pulling |
| 🧭 Import Errors (Windows) | Windows `PYTHONPATH` didn't register correctly | Used `set PYTHONPATH=./app` or ran from project root |
| 🐍 JSONDecodeError | Yahoo blocked tickers returned empty responses | Used `.iloc[0]` properly to fix float casting issue |

---

## ✅ Final Status

- Local test using `test_top_movers.py`: **Passed**
- VM test with fallback logic: **Passed**
- Endpoint `/top-movers`: **Available**
- Code pushed to GitHub

---

## 📌 Future Enhancements

- Move TICKERS to a CSV or DB source
- Add caching to reduce repeated API calls
- Support custom ticker input via query param (e.g. `/top-movers?ticker=...`)

---
