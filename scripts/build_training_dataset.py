import pandas as pd
import os


OBS = "data/raw/mythenquai_zurich.csv"
FORECAST = "data/raw/openmeteo_icon_d2_zurich.csv"

OUTPUT = "data/processed/training.csv"


def load():

    obs = pd.read_csv(OBS)
    forecast = pd.read_csv(FORECAST)

    obs["time"] = pd.to_datetime(obs["time"])
    forecast["time"] = pd.to_datetime(forecast["time"])

    return obs, forecast



def prepare(obs, forecast):

    # convert 10-minute observations to hourly
    obs_hourly = (
        obs
        .set_index("time")
        .resample("1h")
        .mean()
        .reset_index()
    )


    # merge closest hour
    df = pd.merge(
        forecast,
        obs_hourly,
        on="time",
        how="inner"
    )


    # forecast error
    df["wind_error"] = (
        df["wind"]
        -
        df["wind_speed_10m"]
    )


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


    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    df.to_csv(
        OUTPUT,
        index=False
    )


    print("saved:")
    print(OUTPUT)