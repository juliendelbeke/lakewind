import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA = "data/processed/training.csv"
MODEL = "models/baseline_lightgbm.pkl"


def load():

    df = pd.read_csv(DATA)

    df["time"] = pd.to_datetime(df["time"])

    return df



def prepare_test(df):

    # same split as training
    train_end = pd.Timestamp("2026-03-31 23:00")

    test = df[
        df.time > train_end
    ].copy()

    return test



def features(df):

    exclude = [
        "time",
        "wind",
        "wind_error"
    ]

    return [
        c for c in df.columns
        if c not in exclude
    ]



def evaluate_group(
    df,
    group,
    name
):

    print()
    print("==============================")
    print(name)
    print("==============================")

    result = (
        df.groupby(group)
        .apply(
            lambda x: pd.Series({
                "samples": len(x),
                "raw_mae": mean_absolute_error(
                    x.wind,
                    x.wind_speed_10m
                ),
                "lakewind_mae": mean_absolute_error(
                    x.wind,
                    x.prediction
                ),
                "bias": (
                    x.prediction - x.wind
                ).mean()
            })
        )
    )

    print(result)



def main():

    df = load()

    test = prepare_test(df)


    model = joblib.load(MODEL)


    X = test[features(test)]


    test["prediction"] = (
        test.wind_speed_10m
        +
        model.predict(X)
    )


    print("==============================")
    print("GLOBAL PERFORMANCE")
    print("==============================")


    print(
        "samples:",
        len(test)
    )


    print(
        "Raw ICON MAE:",
        round(
            mean_absolute_error(
                test.wind,
                test.wind_speed_10m
            ),
            3
        )
    )


    print(
        "LakeWind MAE:",
        round(
            mean_absolute_error(
                test.wind,
                test.prediction
            ),
            3
        )
    )


    print(
        "RMSE:",
        round(
            mean_squared_error(
                test.wind,
                test.prediction,
                squared=False
            ),
            3
        )
    )


    print(
        "R2:",
        round(
            r2_score(
                test.wind,
                test.prediction
            ),
            3
        )
    )



    # sailing hours

    sailing = test[
        (test.time.dt.hour >= 9)
        &
        (test.time.dt.hour <= 21)
    ]


    print()

    print("==============================")
    print("SAILING WINDOW 09-21")
    print("==============================")


    print(
        "samples:",
        len(sailing)
    )


    print(
        "MAE:",
        round(
            mean_absolute_error(
                sailing.wind,
                sailing.prediction
            ),
            3
        )
    )


    # add diagnostics

    test["hour"] = test.time.dt.hour

    evaluate_group(
        test,
        "hour",
        "ERROR BY HOUR"
    )


    # wind bins

    test["wind_bin"] = pd.cut(
        test.wind,
        bins=[
            -1,
            2,
            4,
            6,
            10,
            20
        ]
    )


    evaluate_group(
        test,
        "wind_bin",
        "ERROR BY OBSERVED WIND"
    )



    # direction

    test["direction_bin"] = pd.cut(
        test.wind_direction_10m,
        bins=[
            0,
            45,
            90,
            135,
            180,
            225,
            270,
            315,
            360
        ]
    )


    evaluate_group(
        test,
        "direction_bin",
        "ERROR BY FORECAST DIRECTION"
    )



    # worst errors

    print()

    print("==============================")
    print("WORST MISSES")
    print("==============================")


    test["error"] = abs(
        test.prediction -
        test.wind
    )


    print(
        test[
            [
                "time",
                "wind",
                "wind_speed_10m",
                "prediction",
                "error"
            ]
        ]
        .sort_values(
            "error",
            ascending=False
        )
        .head(20)
    )



    # feature importance

    print()

    print("==============================")
    print("FEATURE IMPORTANCE")
    print("==============================")


    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    )


    print(
        importance
        .sort_values(
            ascending=False
        )
        .head(30)
    )



if __name__ == "__main__":
    main()