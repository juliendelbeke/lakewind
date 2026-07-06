import requests
import pandas as pd
import os


LAT = 47.36
LON = 8.54

MODEL = "icon_d2"

START_DATE = "2026-06-06"
END_DATE = "2026-07-06"

OUTPUT = "data/raw/openmeteo_icon_d2_zurich.csv"


URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


PARAMS = {
    "latitude": LAT,
    "longitude": LON,

    "hourly": ",".join([
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "temperature_2m",
        "pressure_msl",
        "cloud_cover",
        "precipitation"
    ]),

    "models": MODEL,

    "start_date": START_DATE,
    "end_date": END_DATE,

    "timezone": "Europe/Zurich"
}


def fetch():

    print("Downloading Open-Meteo forecast...")

    r = requests.get(
        URL,
        params=PARAMS
    )

    print(r.url)

    r.raise_for_status()

    return r.json()



def parse(data):

    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    df["time"] = pd.to_datetime(
        df["time"]
    )

    return df



if __name__ == "__main__":

    data = fetch()

    df = parse(data)


    print(df.head())
    print()
    print("rows:", len(df))


    os.makedirs(
        "data/raw",
        exist_ok=True
    )


    df.to_csv(
        OUTPUT,
        index=False
    )


    print()
    print("saved:")
    print(OUTPUT)