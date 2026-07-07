import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error


# ----------------------------
# Load data
# ----------------------------

df = pd.read_csv(
    "data/processed/training.csv",
    parse_dates=["time"]
)


# ----------------------------
# Create same features as model
# ----------------------------

df["hour"] = df.time.dt.hour
df["month"] = df.time.dt.month

df["hour_sin"] = np.sin(
    2 * np.pi * df.hour / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df.hour / 24
)

df["dir_sin"] = np.sin(
    np.deg2rad(df.wind_direction_10m)
)

df["dir_cos"] = np.cos(
    np.deg2rad(df.wind_direction_10m)
)


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


df = df.dropna(
    subset=features + [
        "wind",
        "wind_speed_10m"
    ]
)


# ----------------------------
# Use unseen 2026 period only
# ----------------------------

df = df[
    df.time >= "2026-01-01"
]


# ----------------------------
# Load model
# ----------------------------

model = joblib.load(
    "models/baseline_lightgbm.pkl"
)


df["predicted_error"] = model.predict(
    df[features]
)


df["corrected_wind"] = (
    df["wind_speed_10m"]
    +
    df["predicted_error"]
)


# ----------------------------
# Sailing hours
# ----------------------------

sailing = df[
    (df.time.dt.hour >= 9)
    &
    (df.time.dt.hour <= 21)
]


print()
print("======================")
print("SAILING HOURS 09-21")
print("======================")

print(
    "Rows:",
    len(sailing)
)


print(
    "Raw ICON MAE:",
    round(
        mean_absolute_error(
            sailing.wind,
            sailing.wind_speed_10m
        ),
        3
    )
)


print(
    "LakeWind MAE:",
    round(
        mean_absolute_error(
            sailing.wind,
            sailing.corrected_wind
        ),
        3
    )
)


# ----------------------------
# Error by actual wind regime
# ----------------------------

print()
print("======================")
print("ERROR BY WIND SPEED")
print("======================")


bins = [
    0,
    2,
    4,
    6,
    8,
    12,
    100
]

labels = [
    "0-2 m/s",
    "2-4 m/s",
    "4-6 m/s",
    "6-8 m/s",
    "8-12 m/s",
    ">12 m/s"
]


sailing["wind_range"] = pd.cut(
    sailing.wind,
    bins=bins,
    labels=labels
)


for name, group in sailing.groupby(
    "wind_range",
    observed=True
):

    if len(group) < 20:
        continue

    raw = mean_absolute_error(
        group.wind,
        group.wind_speed_10m
    )

    corrected = mean_absolute_error(
        group.wind,
        group.corrected_wind
    )

    print()
    print(name)
    print("samples:", len(group))
    print("raw:", round(raw,3))
    print("LakeWind:", round(corrected,3))


# ----------------------------
# Error by hour
# ----------------------------

print()
print("======================")
print("ERROR BY HOUR")
print("======================")


hour_stats = sailing.groupby(
    sailing.time.dt.hour
).apply(
    lambda x: pd.Series({
        "raw_mae": mean_absolute_error(
            x.wind,
            x.wind_speed_10m
        ),
        "lakewind_mae": mean_absolute_error(
            x.wind,
            x.corrected_wind
        ),
        "samples": len(x)
    }),
    include_groups=False
)


print(hour_stats)


# ----------------------------
# Strong wind events
# ----------------------------

print()
print("======================")
print("STRONG WIND EVENTS")
print("======================")


strong = sailing[
    sailing.wind >= 6
]


print(
    "Samples:",
    len(strong)
)


if len(strong):

    print(
        "Raw MAE:",
        round(
            mean_absolute_error(
                strong.wind,
                strong.wind_speed_10m
            ),
            3
        )
    )

    print(
        "LakeWind MAE:",
        round(
            mean_absolute_error(
                strong.wind,
                strong.corrected_wind
            ),
            3
        )
    )

df["persistence_error"] = (
    df["wind"] - df["wind_lag_1h"]
)

print(
    "Persistence MAE:",
    df.persistence_error.abs().mean()
)