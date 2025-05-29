# services/backtest.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from typing import Dict
from services.data import get_clean_data
from services.models import load_model_by_name


def backtest(ticker: str, model_name: str, days: int = 30) -> Dict[str, float]:
    try:
        # Load historical stock data
        df = get_clean_data(ticker, period=f"{days * 2}d")
        if df is None or df.empty:
            return {"error": "No data available for this ticker."}

        model = load_model_by_name(model_name)
        if model is None:
            return {"error": "Invalid model name."}

        # Feature Engineering - adjust based on your existing pipeline
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df = df.dropna()

        X = df[["SMA_10", "SMA_20"]]
        y = df["Close"]

        X_train = X.iloc[:-days]
        y_train = y.iloc[:-days]
        X_test = X.iloc[-days:]
        y_test = y.iloc[-days:]

        if hasattr(model, "fit"):  # e.g., Linear model
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        directional_accuracy = np.mean(np.sign(np.diff(y_test.values)) == np.sign(np.diff(y_pred))) * 100

        return {
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "directional_accuracy": round(directional_accuracy, 2)
        }

    except Exception as e:
        return {"error": str(e)}
