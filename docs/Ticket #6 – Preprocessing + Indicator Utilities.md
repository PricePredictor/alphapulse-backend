# ✅ Ticket #6 – Preprocessing + Indicator Utilities

### 📝 Description
Created reusable preprocessing and indicator utility functions for use in all models. Includes:
- `SMA`, `RSI` from price series
- `MinMaxScaler` normalization
- Fallback handling for empty inputs or errors

---

### 📦 Files Added

- `app/utils/indicators.py`
- `app/utils/preprocessing.py`
- `scripts/test_utils.py`

---

### 📁 indicators.py Functions

#### `calculate_sma(series: pd.Series, window: int) -> pd.Series`
Returns Simple Moving Average over `window` periods.  
Fallback: Returns `NaN` series if error.

#### `calculate_rsi(series: pd.Series, window: int) -> pd.Series`
Returns Relative Strength Index using `ta` lib.  
Fallback: Returns `NaN` series if error.

---

### 📁 preprocessing.py Functions

#### `normalize_dataframe(df: pd.DataFrame) -> np.ndarray`
Scales each column to 0–1 using `MinMaxScaler`.  
Fallback: Returns zeros matrix if DataFrame is empty or error occurs.

---

### ✅ Test Scenarios

| Test Case             | How to Run                         | Expected Output                         |
|-----------------------|-------------------------------------|------------------------------------------|
| **SMA works**         | Run `calculate_sma(data)`          | Series of moving averages               |
| **RSI works**         | Run `calculate_rsi(data)`          | RSI values (0–100)                      |
| **Normalize works**   | Run `normalize_dataframe(df)`      | Numpy array scaled between 0–1          |
| **Empty fallback**    | Run `normalize_dataframe(empty_df)`| Zero array with same shape as input     |

---

### 🧪 Validation

Run via:
```bash
PYTHONPATH=./app python3 scripts/test_utils.py
