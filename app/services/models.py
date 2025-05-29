# services/models.py

from sklearn.linear_model import LinearRegression
import joblib
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model as load_keras_model

def load_model_by_name(name: str):
    if name == "linear":
        return LinearRegression()
    elif name == "xgb":
        return joblib.load("models/xgb_model.pkl")
    elif name == "lstm":
        return load_keras_model("models/lstm_model.h5")
    else:
        return None
