"""Quick smoke test for prediction.py"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from prediction import (
    check_prediction_suitability,
    train_regressors,
    compare_regressors,
    predict_value,
    get_prediction_disclaimer,
)

np.random.seed(0)
n = 100
df = pd.DataFrame(
    {
        "temp": np.random.uniform(10, 30, n),
        "salinity": np.random.uniform(30, 38, n),
        "depth": np.random.uniform(0, 200, n),
        "pollution": np.random.uniform(0, 50, n),
    }
)

# 1. suitability - happy path
suit = check_prediction_suitability(df, "pollution")
assert suit["suitable"], "Expected suitable=True"
print("Suitability OK:", suit["suitable"])

# edge: non-numeric target
df2 = df.copy()
df2["pollution"] = "bad"
assert not check_prediction_suitability(df2, "pollution")["suitable"]
print("Non-numeric catch OK")

# edge: zero-variance
df3 = df.copy()
df3["pollution"] = 5.0
assert not check_prediction_suitability(df3, "pollution")["suitable"]
print("Zero-variance catch OK")

# edge: missing column
assert not check_prediction_suitability(df, "nonexistent")["suitable"]
print("Missing column catch OK")

# 2. train_regressors
feature_cols = ["temp", "salinity", "depth"]
results = train_regressors(df, feature_cols, "pollution", test_size=0.2, random_state=42)
print("Models trained:", list(results.keys()))
for name, info in results.items():
    assert info["model"] is not None, f"{name} model is None"
    mae, rmse, r2 = info["mae"], info["rmse"], info["r2"]
    fi = info["feature_importance"]
    print(f"  {name}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  fi_keys={list(fi.keys())}")

# time-split variant
results_ts = train_regressors(df, feature_cols, "pollution", use_time_split=True)
print("Time-split OK:", list(results_ts.keys()))

# 3. compare_regressors
cmp = compare_regressors(results)
print(cmp.to_string(index=False))
assert list(cmp.columns) == ["Model", "MAE", "RMSE", "R2"]
print("compare_regressors OK")

# empty dict edge case
empty_cmp = compare_regressors({})
assert list(empty_cmp.columns) == ["Model", "MAE", "RMSE", "R2"]
print("compare_regressors empty dict OK")

# 4. predict_value
lr_model = results["LinearRegression"]["model"]
val = predict_value(lr_model, feature_cols, {"temp": 20.0, "salinity": 34.5, "depth": 50.0})
print(f"Single prediction: {val:.4f}")
assert isinstance(val, float)

# edge: missing feature key
try:
    predict_value(lr_model, feature_cols, {"temp": 20.0})
    raise AssertionError("Should have raised ValueError")
except ValueError as e:
    print("Missing feature caught:", e)

# edge: None model
try:
    predict_value(None, feature_cols, {"temp": 20.0, "salinity": 34.5, "depth": 50.0})
    raise AssertionError("Should have raised ValueError")
except ValueError as e:
    print("None model caught:", e)

# 5. disclaimer
disc = get_prediction_disclaimer()
assert len(disc) > 50
print("Disclaimer OK, length:", len(disc))

print()
print("All checks PASSED")
