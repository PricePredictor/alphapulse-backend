
# AlphaPulse Backend

This is a FastAPI backend for the AlphaPulse stock prediction platform.

## Endpoints

- `/` — Health check
- `/live-price?symbol=AAPL` — Get real-time stock price
- `/predict?symbol=AAPL` — Get a dummy AI prediction

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
