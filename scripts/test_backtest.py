# test_backtest.py

from services.backtest import backtest

# Test with valid inputs
print("Valid case:", backtest("AAPL", "xgb", days=30))

# Test with fake ticker
print("Invalid ticker:", backtest("FAKE123", "xgb", days=30))

# Test with bad model
print("Invalid model:", backtest("AAPL", "fake_model", days=30))
