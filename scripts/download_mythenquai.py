import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

BASE = "https://www.tecson-data.ch"
INDEX = BASE + "/zurich/mythenquai/index.php"
URL = BASE + "/zurich/mythenquai/uebersicht/messwerte.php"

payload = {
    "messw_beg": "06.06.2023",
    "messw_end": "06.07.2026",
    "felder[]": ["WGmax", "WGavr", "WRvek"],
    "auswahl": "2",
    "combilog": "mythenquai",
    "suchen": "Werte anzeigen"
}


def fetch():
    session = requests.Session()

    session.get(
        INDEX,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    r = session.post(
        URL,
        data=payload,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": INDEX,
            "Origin": BASE
        }
    )

    print("status:", r.status_code)

    r.raise_for_status()

    return r.text


def parse(html):
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table", {"border": "1"})

    if table is None:
        raise Exception("Could not find data table")

    rows = table.find_all("tr")

    data = []

    for row in rows[1:]:
        cols = row.find_all("td")

        if len(cols) != 4:
            continue

        data.append([
            cols[0].get_text(strip=True),
            cols[1].get_text(strip=True),
            cols[2].get_text(strip=True),
            cols[3].get_text(strip=True),
        ])

    df = pd.DataFrame(
        data,
        columns=[
            "time",
            "gust",
            "wind",
            "direction"
        ]
    )

    df["time"] = pd.to_datetime(
        df["time"],
        dayfirst=True,
        errors="coerce"
    )

    for col in ["wind", "gust", "direction"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()

    return df


if __name__ == "__main__":

    html = fetch()

    df = parse(html)

    print(df.head())
    print()
    print("rows:", len(df))

    os.makedirs("data", exist_ok=True)

    df.to_csv(
        "data/raw/mythenquai_zurich1.csv",
        index=False
    )

    print("saved")