# ✅ Ticket #8 – Closing Documentation

**Title:**  
`[Endpoint] Implement /predict-ensemble Endpoint`

**Sprint:**  
Week 1 – June 1–7

**Milestone:**  
Phase 1 – Model Integration

---

## 📝 Description
Expose `/predict-ensemble` endpoint that runs multiple models, compares outputs, and returns a consensus or average prediction.

---

## 📦 Deliverables

- ✅ `GET /predict-ensemble?ticker=AAPL` endpoint exposed via FastAPI
- ✅ Aggregates predictions from:
  - XGBoost
  - LSTM
  - Linear Regression
- ✅ Returns:
```json
{
  "ticker": "AAPL",
  "ensemble": 133.25,
  "models": {
    "xgb": 132.4,
    "lstm": 134.1,
    "linear": 133.25
  }
}
```

---

## ✅ Acceptance Criteria

| Test Scenario         | Method                              | Expected Result                            | ✅ Status |
|----------------------|--------------------------------------|---------------------------------------------|----------|
| Valid ticker         | `/predict-ensemble?ticker=AAPL`      | JSON with average and breakdown             | ✅ Passed |
| Missing ticker       | `/predict-ensemble`                  | 400 with "Ticker is required" error         | ✅ Passed |
| Output format        | JSON response                        | Includes `ticker`, `ensemble`, `models`     | ✅ Passed |
| One model fails      | Simulated exception in one model     | Still returns result from others            | ✅ Passed |

---

## 🛠 Tools Used

- FastAPI  
- Python dictionary operations  
- Docker (forced rebuild to ensure updated code is used)  
- Swagger UI (`/docs`) for validation

---

## 🧱 Related Tickets

- ✅ #5: Modular Model Pipelines (XGBoost, LSTM, Linear)  
- ✅ #6: Preprocessing + Indicator Utilities  
- ⏳ #10 (Upcoming): Backtest Utility for Single Model  
- 🔄 #9 (Upcoming): Top Movers – Daily % Change Analysis

---

## 🧪 Dummy Functions Implemented

To simulate model behavior during development, the following mock functions were used:

- `app/models/xgb_model.py`:
```python
def predict_xgb(ticker: str) -> float:
    return 132.4
```

- `app/models/lstm_model.py`:
```python
def predict_lstm(ticker: str) -> float:
    return 134.1
```

- `app/models/linear_model.py`:
```python
def predict_linear(ticker: str) -> float:
    return 133.25
```

These will be replaced with actual model inference logic in future phases.

---

## 🚀 Final Result

Live and successfully tested at:

```
http://135.232.111.213:8000/predict-ensemble?ticker=AAPL
```

The endpoint:
- Returns expected ensemble JSON
- Handles missing or partial model failures gracefully
- Is documented in Swagger
- Is deployed and verified inside the Dockerized production environment