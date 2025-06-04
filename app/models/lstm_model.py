import numpy as np
from tensorflow.keras.models import load_model

def predict(ticker: str, data):
    try:
        model = load_model("app/models/lstm_model.h5")
        reshaped = data.values.reshape((1, data.shape[0], data.shape[1]))
        prediction = model.predict(reshaped)
        return float(prediction[0][0])
    except Exception as e:
        print(f"[LSTM] Error: {e}")
        return np.nan
