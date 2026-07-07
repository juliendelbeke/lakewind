import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import joblib


df = pd.read_csv(
    "data/processed/training.csv",
    parse_dates=["time"]
)


# ----------------------------
# Time features
# ----------------------------

df["hour"] = df.time.dt.hour
df["month"] = df.time.dt.month

df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)

df["dir_sin"] = np.sin(
    np.deg2rad(df.wind_direction_10m)
)

df["dir_cos"] = np.cos(
    np.deg2rad(df.wind_direction_10m)
)


# ----------------------------
# Predict the correction
# ----------------------------

target = "wind_error"


features = [
    "wind_gusts_10m",
    "temperature_2m",
    "pressure_msl",
    "cloud_cover",
    "precipitation",
    "hour_sin",
    "hour_cos",
    "month",
    "dir_sin",
    "dir_cos",
]


# remove incomplete rows

df = df.dropna(
    subset=features + [
        target,
        "wind",
        "wind_speed_10m"
    ]
)


# ----------------------------
# Chronological split
# ----------------------------

train = df[
    df["time"] < "2026-01-01"
]

test = df[
    df["time"] >= "2026-01-01"
]


print("Train rows:", len(train))
print("Test rows:", len(test))

print()
print(
    "Train period:",
    train.time.min(),
    "to",
    train.time.max()
)

print(
    "Test period:",
    test.time.min(),
    "to",
    test.time.max()
)


X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


baseline_forecast = test["wind_speed_10m"]
observed = test["wind"]


# ----------------------------
# Train
# ----------------------------

model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


# ----------------------------
# Constant bias benchmark
# ----------------------------

constant_bias = y_train.mean()

constant_prediction = (
    baseline_forecast + constant_bias
)


constant_mae = mean_absolute_error(
    observed,
    constant_prediction
)


print()
print(
    "Constant bias MAE:",
    round(constant_mae, 3)
)


# ----------------------------
# Predict correction
# ----------------------------

predicted_error = model.predict(
    X_test
)


corrected_forecast = (
    baseline_forecast + predicted_error
)


# ----------------------------
# Evaluate
# ----------------------------

baseline_mae = mean_absolute_error(
    observed,
    baseline_forecast
)


corrected_mae = mean_absolute_error(
    observed,
    corrected_forecast
)


print()
print(
    "Baseline MAE :",
    round(baseline_mae, 3)
)

print(
    "Corrected MAE:",
    round(corrected_mae, 3)
)


improvement = (
    100 *
    (baseline_mae - corrected_mae)
    /
    baseline_mae
)


print()
print(
    "Improvement:",
    round(improvement, 2),
    "%"
)


# ----------------------------
# Feature importance
# ----------------------------

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(
    ascending=False
)


print()
print(importance)


# ----------------------------
# Save model
# ----------------------------

joblib.dump(
    model,
    "models/baseline_lightgbm.pkl"
)


print()
print("Model saved.")