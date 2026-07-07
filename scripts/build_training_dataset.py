import pandas as pd
import numpy as np
import os


OBS = "data/raw/mythenquai_zurich.csv"
FORECAST = "data/raw/openmeteo_icon_ch1_zurich.csv"

OUTPUT = "data/processed/training.csv"



def load():

    obs = pd.read_csv(OBS)
    forecast = pd.read_csv(FORECAST)

    obs["time"] = pd.to_datetime(obs["time"])
    forecast["time"] = pd.to_datetime(forecast["time"])

    return obs, forecast



def prepare(obs, forecast):

    # ----------------------------
    # Convert 10-min observations to hourly
    # ----------------------------

    obs_hourly = (
        obs
        .set_index("time")
        .resample("1h")
        .mean()
        .reset_index()
    )


    # ----------------------------
    # Merge forecast + observation
    # ----------------------------

    df = pd.merge(
        forecast,
        obs_hourly,
        on="time",
        how="inner"
    )


    df = df.sort_values("time").reset_index(drop=True)


    # ----------------------------
    # Target
    # ----------------------------

    df["wind_error"] = (
        df["wind"]
        -
        df["wind_speed_10m"]
    )


    # ----------------------------
    # Previous station state
    # ----------------------------

    df["wind_lag_1h"] = df["wind"].shift(1)
    df["wind_lag_2h"] = df["wind"].shift(2)
    df["wind_lag_3h"] = df["wind"].shift(3)

    df["gust_lag_1h"] = df["gust"].shift(1)

    df["direction_lag_1h"] = df["direction"].shift(1)


    # Wind evolution

    df["wind_change_1h"] = (
        df["wind_lag_1h"]
        -
        df["wind_lag_2h"]
    )


    df["wind_change_3h"] = (
        df["wind_lag_1h"]
        -
        df["wind_lag_3h"]
    )


    # ----------------------------
    # Time features
    # ----------------------------

    df["hour"] = df.time.dt.hour
    df["month"] = df.time.dt.month
    df["day_of_year"] = df.time.dt.dayofyear


    df["hour_sin"] = np.sin(
        2*np.pi*df.hour/24
    )

    df["hour_cos"] = np.cos(
        2*np.pi*df.hour/24
    )


    df["doy_sin"] = np.sin(
        2*np.pi*df.day_of_year/365
    )

    df["doy_cos"] = np.cos(
        2*np.pi*df.day_of_year/365
    )



    # ----------------------------
    # Forecast wind direction
    # ----------------------------

    radians = np.deg2rad(
        df["wind_direction_10m"]
    )


    df["dir_sin"] = np.sin(radians)
    df["dir_cos"] = np.cos(radians)


    df["u10"] = (
        df.wind_speed_10m *
        np.sin(radians)
    )

    df["v10"] = (
        df.wind_speed_10m *
        np.cos(radians)
    )



    # ----------------------------
    # Gust features
    # ----------------------------

    df["gust_difference"] = (
        df.wind_gusts_10m -
        df.wind_speed_10m
    )



    # ----------------------------
    # Radiation / thermal features
    # ----------------------------

    df["radiation_total"] = (
        df["shortwave_radiation"]
        +
        df["direct_radiation"]
        +
        df["diffuse_radiation"]
    )


    df["rain_3h"] = (
        df["precipitation"]
        .rolling(3)
        .sum()
    )


    df["temp_dew_spread"] = (
        df["temperature_2m"]
        -
        df["dew_point_2m"]
    )


    # ----------------------------
    # Direction classes
    # ----------------------------

    df["is_west"] = (
        (df.wind_direction_10m >= 240) &
        (df.wind_direction_10m <= 330)
    ).astype(int)


    df["is_north"] = (
        df.wind_direction_10m <= 60
    ).astype(int)


    df["is_south"] = (
        (df.wind_direction_10m >= 150) &
        (df.wind_direction_10m <= 230)
    ).astype(int)



    # ----------------------------
    # Lake breeze proxies
    # ----------------------------

    df["afternoon"] = (
        (df.hour >= 12) &
        (df.hour <= 18)
    ).astype(int)


    df["summer"] = (
        df.month.isin(
            [5,6,7,8,9]
        )
    ).astype(int)


    df["summer_afternoon"] = (
        (df.summer == 1) &
        (df.afternoon == 1)
    ).astype(int)



    # Remove empty model rows

    df = df.dropna()


    return df



if __name__ == "__main__":

    obs, forecast = load()

    df = prepare(
        obs,
        forecast
    )


    print(df.head())
    print()
    print("rows:", len(df))
    print()
    print("columns:")
    print(list(df.columns))


    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    df.to_csv(
        OUTPUT,
        index=False
    )


    print()
    print("saved:")
    print(OUTPUT)