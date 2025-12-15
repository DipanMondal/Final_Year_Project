import argparse
import sqlite3
from pathlib import Path
import json

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DB_PATH = Path("data/weather.db")

def upsert_csv_to_db(city: str, csv_path: str):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    rows = [(city, r["date"], float(r["tmin"]), float(r["tmax"]), float(r["tavg"])) for _, r in df.iterrows()]
    cur.executemany(
        "INSERT OR REPLACE INTO weather_daily (city,date,tmin,tmax,tavg) VALUES (?,?,?,?,?)",
        rows
    )
    con.commit()
    con.close()
    return len(df)

def load_city_series(city: str):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT date,tavg FROM weather_daily WHERE city=? ORDER BY date ASC",
        con,
        params=(city,)
    )
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()
    return df

def build_features(df: pd.DataFrame):
    # df: columns [date, tavg]
    x = df.copy()
    x["tavg_lag_1"] = x["tavg"].shift(1)
    x["tavg_lag_2"] = x["tavg"].shift(2)
    x["tavg_lag_7"] = x["tavg"].shift(7)
    x["tavg_roll_mean_7"] = x["tavg"].shift(1).rolling(7).mean()
    x["tavg_roll_std_7"] = x["tavg"].shift(1).rolling(7).std()

    doy = x["date"].dt.dayofyear
    x["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # target: next-day temperature
    x["y"] = x["tavg"].shift(-1)

    x = x.dropna().reset_index(drop=True)
    feature_cols = ["tavg_lag_1","tavg_lag_2","tavg_lag_7","tavg_roll_mean_7","tavg_roll_std_7","doy_sin","doy_cos"]
    return x[feature_cols], x["y"], feature_cols, x

def time_split(full: pd.DataFrame, train_end: str, val_end: str):
    # train <= train_end, val (train_end, val_end], test > val_end
    d = full.copy()
    train = d[d["date"] <= pd.to_datetime(train_end)]
    val = d[(d["date"] > pd.to_datetime(train_end)) & (d["date"] <= pd.to_datetime(val_end))]
    test = d[d["date"] > pd.to_datetime(val_end)]
    return train, val, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--csv", required=True, help="Path to ingested CSV")
    ap.add_argument("--train_end", default="2022-12-31")
    ap.add_argument("--val_end", default="2023-12-31")
    args = ap.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    n = upsert_csv_to_db(args.city, args.csv)
    print(f"Loaded {n} rows into DB for city={args.city}")

    series = load_city_series(args.city)
    X, y, feature_cols, full_feat = build_features(series)

    # keep the date column for splitting
    full = full_feat[["date"]].copy()
    for c in feature_cols:
        full[c] = X[c].values
    full["y"] = y.values

    train, val, test = time_split(full, args.train_end, args.val_end)

    X_train, y_train = train[feature_cols], train["y"]
    X_val, y_val = val[feature_cols], val["y"]
    X_test, y_test = test[feature_cols], test["y"]

    model = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        max_depth=None
    )
    model.fit(X_train, y_train)

    def eval_block(name, Xb, yb):
        pred = model.predict(Xb)
        mae = float(mean_absolute_error(yb, pred))
        rmse = float(np.sqrt(mean_squared_error(yb, pred)))
        print(f"{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  n={len(yb)}")
        return mae, rmse

    val_mae, val_rmse = eval_block("VAL", X_val, y_val)
    test_mae, test_rmse = eval_block("TEST", X_test, y_test)

    dump(model, "artifacts/temp_model.joblib")

    with open("artifacts/feature_config.json", "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols}, f, indent=2)

    with open("artifacts/train_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "city": args.city,
            "train_end": args.train_end,
            "val_end": args.val_end,
            "metrics": {"val_mae": val_mae, "val_rmse": val_rmse, "test_mae": test_mae, "test_rmse": test_rmse}
        }, f, indent=2)

    print("Saved artifacts in /artifacts")

if __name__ == "__main__":
    main()
