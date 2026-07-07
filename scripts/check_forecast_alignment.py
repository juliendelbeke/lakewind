import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error


FILE = "data/processed/training.csv"


df = pd.read_csv(
    FILE,
    parse_dates=["time"]
)


# ---------------------------------------
# Clean
# ---------------------------------------

df = df.dropna(
    subset=[
        "wind",
        "wind_speed_10m",
        "wind_direction_10m"
    ]
)


obs = df["wind"]
icon = df["wind_speed_10m"]


print("==============================")
print("BASIC SCALING")
print("==============================")


# Optimal multiplicative correction
scale = np.sum(obs * icon) / np.sum(icon ** 2)

print(
    "Optimal scale factor:",
    round(scale,3)
)


scaled = icon * scale


print(
    "Raw ICON MAE:",
    round(mean_absolute_error(obs, icon),3)
)

print(
    "Scaled ICON MAE:",
    round(mean_absolute_error(obs, scaled),3)
)


print(
    "Raw bias:",
    round((obs-icon).mean(),3)
)

print(
    "Scaled bias:",
    round((obs-scaled).mean(),3)
)



# ---------------------------------------
# Direction dependent scaling
# ---------------------------------------

print()
print("==============================")
print("SCALING BY WIND DIRECTION")
print("==============================")


df["direction_bin"] = pd.cut(
    df.wind_direction_10m,
    bins=np.arange(0,361,45),
    include_lowest=True
)


direction = (
    df.groupby(
        "direction_bin",
        observed=True
    )
    .apply(
        lambda x: pd.Series({
            "samples":len(x),

            "scale":
                np.sum(
                    x.wind*x.wind_speed_10m
                )
                /
                np.sum(
                    x.wind_speed_10m**2
                ),

            "raw_mae":
                mean_absolute_error(
                    x.wind,
                    x.wind_speed_10m
                ),

            "scaled_mae":
                mean_absolute_error(
                    x.wind,
                    x.wind_speed_10m *
                    (
                    np.sum(
                    x.wind*x.wind_speed_10m
                    )
                    /
                    np.sum(
                    x.wind_speed_10m**2
                    )
                    )
                )
        })
    )
)


print(direction)



# ---------------------------------------
# Hourly scaling
# ---------------------------------------

print()
print("==============================")
print("SCALING BY HOUR")
print("==============================")


df["hour"] = df.time.dt.hour


hour = (
    df.groupby("hour")
    .apply(
        lambda x: pd.Series({

            "samples":len(x),

            "scale":
                np.sum(
                    x.wind*x.wind_speed_10m
                )
                /
                np.sum(
                    x.wind_speed_10m**2
                ),

            "mae":
                mean_absolute_error(
                    x.wind,
                    x.wind_speed_10m
                )
        })
    )
)


print(hour)



# ---------------------------------------
# Plot observed vs scaled
# ---------------------------------------

plt.figure(figsize=(6,6))

plt.scatter(
    icon,
    obs,
    alpha=0.2,
    label="Raw"
)

plt.scatter(
    scaled,
    obs,
    alpha=0.2,
    label="Scaled"
)

plt.xlabel(
    "Forecast wind (m/s)"
)

plt.ylabel(
    "Observed wind (m/s)"
)

plt.legend()

plt.grid()

plt.title(
    "ICON scaling correction"
)

plt.savefig(
    "data/processed/icon_scaling.png",
    dpi=150
)



# ---------------------------------------
# Direction plot
# ---------------------------------------

plt.figure(figsize=(8,4))

plt.bar(
    direction.index.astype(str),
    direction["scale"]
)

plt.xticks(
    rotation=45
)

plt.ylabel(
    "Observed / ICON scaling factor"
)

plt.title(
    "Wind speed scaling by direction"
)

plt.tight_layout()

plt.savefig(
    "data/processed/direction_scaling.png",
    dpi=150
)


print()
print("Saved plots:")
print("data/processed/icon_scaling.png")
print("data/processed/direction_scaling.png")