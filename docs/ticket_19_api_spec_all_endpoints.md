Perfect. Here's the starting structure and content for your file:

---

### 📄 File: `docs/ticket_19_api_spec_all_endpoints.md`

````markdown
# 📘 Ticket #19 – Full API Specification (All Endpoints)

This document provides complete reference for all current FastAPI endpoints supported by the AlphaPulse backend, including their purpose, parameters, expected responses, and notes for future extension.

---

## 🔍 1. `/predict`

**Method**: `GET`  
**Description**: Predicts future price for a given stock using the selected model.

**Query Parameters**:
- `ticker` (str) – e.g., `AAPL`
- `model` (str) – options: `xgb`, `lstm`, `linear`

**Response Example**:
```json
{
  "ticker": "AAPL",
  "model": "xgb",
  "prediction": 187.23
}
````

---

## 🧠 2. `/predict-ensemble`

**Method**: `GET`
**Description**: Runs multiple models and returns an ensemble prediction.

**Query Parameters**:

* `ticker` (str) – e.g., `TSLA`

**Response Example**:

```json
{
  "ticker": "TSLA",
  "ensemble": 201.15,
  "models": {
    "xgb": 198.2,
    "lstm": 203.0,
    "linear": 202.25
  }
}
```

---

## 📈 3. `/top-movers`

**Method**: `GET`
**Description**: Returns the top 5 gainers and losers (by % daily change) from a watchlist.

**Response Example**:

```json
{
  "gainers": [
    {"ticker": "AAPL", "change": 3.1},
    ...
  ],
  "losers": [
    {"ticker": "NFLX", "change": -2.5},
    ...
  ]
}
```

---

## 📊 4. `/backtest`

**Method**: `GET`
**Description**: Backtests a prediction model against historical data for a single ticker.

**Query Parameters**:

* `ticker` (str)
* `model` (str)

**Response**:

```json
{
  "ticker": "GOOGL",
  "model": "xgb",
  "rmse": 2.45,
  "mape": 1.23,
  "directional_accuracy": 0.72
}
```

---

## 🔁 5. `/accuracy`

**Method**: `GET`
**Description**: Returns model accuracy metrics for a given ticker.

**Query Parameters**:

* `ticker` (str)
* `model` (str)

**Response**:

```json
{
  "ticker": "MSFT",
  "model": "lstm",
  "rmse": 3.02,
  "mape": 2.1,
  "directional_accuracy": 0.65
}
```

---

## 🩺 6. `/health-check`

**Method**: `GET`
**Description**: Returns health status of the backend system.

**Response**:

```json
{
  "status": "ok"
}
```

---

## 🧪 Future Endpoints (Planned)

* `/predict-hourly`
* `/backtest-multi`
* `/accuracy-multi`
* `/metrics/live`

These will follow similar request/response structure, extended for multi-day/hour formats and model comparisons.

---

## 🛠️ Notes

* All responses are returned in JSON format.
* All endpoints support `CORS` from `*` for now (temporary).
* Model options: `xgb`, `lstm`, `linear` (more coming).
* You can extend this doc to generate OpenAPI or Postman specs later.

````

---

### ✅ What to do next:

1. **Copy the above** content into a new Markdown file:  
   `docs/ticket_19_api_spec_all_endpoints.md`

2. In terminal (from project root):

```bash
git pull origin main
git add docs/ticket_19_api_spec_all_endpoints.md
git commit -m "📘 Ticket #19: Added full API spec documentation for all endpoints"
git push origin main
````

Once done, I’ll prepare the closure summary and move us to the next Sprint 2 ticket. Ready?
