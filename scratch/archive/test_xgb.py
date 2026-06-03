import xgboost as xgb
import numpy as np

def obj(y_true, y_pred):
    return y_pred - y_true, np.ones_like(y_true)

try:
    model = xgb.XGBRegressor(objective=obj, n_estimators=2)
    model.fit(np.array([[1],[2]]), np.array([1,2]))
    print("SUCCESS: Signature is (y_true, y_pred)")
except Exception as e:
    print(f"FAILED: {e}")
