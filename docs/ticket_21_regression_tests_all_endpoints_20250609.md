# ✅ Ticket #21 – Regression Test Pass for All Endpoints

**Date:** 2025-06-09  
**Owner:** Gaurav Malik

---

### 🔍 Tested Endpoints

| Endpoint              | Status | Response Time | Notes                          |
|-----------------------|--------|----------------|--------------------------------|
| `/predict`            | ✅ OK   | ~850 ms        | Returns valid forecast         |
| `/backtest`           | ✅ OK   | ~1.4 s         | No crashes                     |
| `/accuracy`           | ✅ OK   | ~1.2 s         | Rolling metrics consistent     |
| `/top-movers`         | ✅ OK   | ~900 ms        | Sorted gainers & losers list  |
| `/predict-ensemble`   | ✅ OK   | ~1.3 s         | Consensus model returns float |

---

### 🧪 Observations

- All endpoints are returning results in under 2 seconds.
- No `500` errors encountered.
- Ensemble prediction returned consistent results on multiple test tickers.

---

### 📦 Conclusion

✅ Regression pass completed successfully. All endpoints working as expected in both local and staging environments.

