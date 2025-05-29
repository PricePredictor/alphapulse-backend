import joblib
import numpy as np

def predict(ticker: str, data):
    try:
        model = joblib.load(f"app/models/xgb_model.pkl")
        prediction = model.predict(data)
        return prediction[-1] if hasattr(prediction, '__getitem__') else prediction
    except Exception as e:
        print(f"[XGBoost] Error: {e}")
        return np.nan
