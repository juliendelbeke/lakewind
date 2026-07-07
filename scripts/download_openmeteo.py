import requests
import pandas as pd
import os


LAT = 47.3574
LON = 8.5363

START_DATE = "2023-07-06"
END_DATE = "2026-07-06"

MODEL = "meteoswiss_icon_ch1"

OUTPUT = "data/raw/openmeteo_icon_ch1_zurich.csv"


def download():

    url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
    )

    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join([
            # Wind
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",

            # Thermodynamics
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",

            # Pressure / clouds / rain
            "pressure_msl",
            "cloud_cover",
            "precipitation",

            # Radiation (lake breeze proxy)
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",

            # Stability
            "cape"
        ]),
        "models": MODEL,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": "Europe/Zurich"
    }


    print("Downloading Open-Meteo forecast...")
    print(
        requests.Request(
            "GET",
            url,
            params=params
        ).prepare().url
    )


    r = requests.get(
        url,
        params=params
    )

    r.raise_for_status()

    return r.json()



def parse(data):

    df = pd.DataFrame(
        data["hourly"]
    )


    df["time"] = pd.to_datetime(
        df["time"]
    )


    # Remove rows where wind forecast itself is missing
    df = df.dropna(
        subset=[
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "temperature_2m",
            "pressure_msl",
        ]
    )


    # Open-Meteo CH1 returns km/h for wind variables
    # convert to m/s
    df["wind_speed_10m"] = (
        df["wind_speed_10m"] / 3.6
    )

    df["wind_gusts_10m"] = (
        df["wind_gusts_10m"] / 3.6
    )


    return df



if __name__ == "__main__":

    data = download()

    df = parse(data)


    print(df.head())
    print()
    print(df.info())


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