import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_dataframe(df: pd.DataFrame) -> np.ndarray:
    try:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        scaler = MinMaxScaler()
        return scaler.fit_transform(df)
    except Exception as e:
        print(f"[Normalization Error] {e}")
        return np.zeros_like(df.values)
