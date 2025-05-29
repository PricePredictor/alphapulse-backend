from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Create a minimal LSTM model that matches input_shape used in lstm_model.py
model = Sequential()
model.add(LSTM(units=50, input_shape=(10, 4)))  # 10 time steps, 4 features
model.add(Dense(1))  # Single output

model.compile(optimizer="adam", loss="mse")

# Save model to file
model.save("app/models/lstm_model.h5")

print("✅ LSTM model saved to app/models/lstm_model.h5")
