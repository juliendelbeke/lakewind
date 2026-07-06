import pandas as pd
import matplotlib.pyplot as plt


FILE = "data/processed/training.csv"


df = pd.read_csv(FILE)

df["time"] = pd.to_datetime(df["time"])


print("\nBasic statistics")
print("----------------")
print(df["wind_error"].describe())


print("\nMean error:")
print(df["wind_error"].mean())


print("\nError by wind direction")
print("-----------------------")

direction_bins = pd.cut(
    df["wind_direction_10m"],
    bins=8
)

print(
    df.groupby(direction_bins, observed=True)["wind_error"]
    .mean()
)


print("\nError by hour")
print("-------------")

df["hour"] = df["time"].dt.hour

print(
    df.groupby("hour")["wind_error"]
    .mean()
)


# plots

plt.figure(figsize=(8,4))
plt.scatter(
    df["wind_speed_10m"],
    df["wind_error"],
    alpha=0.4
)

plt.xlabel("ICON-D2 wind speed (m/s)")
plt.ylabel("Observed - Forecast (m/s)")
plt.title("Wind forecast error vs forecast wind")

plt.grid()

plt.savefig(
    "data/processed/error_vs_wind.png",
    dpi=150
)


plt.figure(figsize=(8,4))

plt.scatter(
    df["wind_direction_10m"],
    df["wind_error"],
    alpha=0.4
)

plt.xlabel("Forecast wind direction")
plt.ylabel("Error")

plt.title("Wind error vs direction")

plt.grid()

plt.savefig(
    "data/processed/error_vs_direction.png",
    dpi=150
)


print("\nPlots saved")