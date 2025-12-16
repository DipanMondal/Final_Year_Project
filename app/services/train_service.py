from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .db import fetch_history

FEATURE_COLS = ["tavg_lag_1","tavg_lag_2","tavg_lag_7","tavg_roll_mean_7","tavg_roll_std_7","doy_sin","doy_cos"]

def build_features(df: pd.DataFrame):
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values("date")

    x["tavg_lag_1"] = x["tavg"].shift(1)
    x["tavg_lag_2"] = x["tavg"].shift(2)
    x["tavg_lag_7"] = x["tavg"].shift(7)
    x["tavg_roll_mean_7"] = x["tavg"].shift(1).rolling(7).mean()
    x["tavg_roll_std_7"] = x["tavg"].shift(1).rolling(7).std()

    doy = x["date"].dt.dayofyear
    x["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    x["y"] = x["tavg"].shift(-1)
    x = x.dropna().reset_index(drop=True)
    return x

def default_splits(last_date: pd.Timestamp):
    # last 365 days -> test, previous 365 -> val, rest -> train
    test_start = last_date - pd.Timedelta(days=365)
    val_start = last_date - pd.Timedelta(days=730)
    return val_start, test_start

def train_city_model(city_key: str):
    hist = fetch_history(city_key, None, None)
    if hist.empty:
        raise ValueError("No history found in DB for this city.")

    feat = build_features(hist[["date","tavg"]].dropna())
    if len(feat) < 900:
        raise ValueError("Not enough clean data to train. Try a longer date range (3+ years).")

    last_date = pd.to_datetime(feat["date"].iloc[-1])
    val_start, test_start = default_splits(last_date)

    train_df = feat[feat["date"] < val_start]
    val_df   = feat[(feat["date"] >= val_start) & (feat["date"] < test_start)]
    test_df  = feat[feat["date"] >= test_start]

    X_train, y_train = train_df[FEATURE_COLS], train_df["y"]
    X_val, y_val     = val_df[FEATURE_COLS], val_df["y"]
    X_test, y_test   = test_df[FEATURE_COLS], test_df["y"]

    model = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    def metrics(Xb, yb):
        pred = model.predict(Xb)
        mae = float(mean_absolute_error(yb, pred))
        rmse = float(np.sqrt(mean_squared_error(yb, pred)))
        return mae, rmse

    val_mae, val_rmse = metrics(X_val, y_val)
    test_mae, test_rmse = metrics(X_test, y_test)

    out_dir = Path("artifacts/models") / city_key
    out_dir.mkdir(parents=True, exist_ok=True)

    dump(model, out_dir / "temp_model.joblib")
    (out_dir / "feature_config.json").write_text(json.dumps({"feature_cols": FEATURE_COLS}, indent=2), encoding="utf-8")
    (out_dir / "train_metadata.json").write_text(json.dumps({
        "city": city_key,
        "split": {"val_start": str(val_start.date()), "test_start": str(test_start.date())},
        "metrics": {"val_mae": val_mae, "val_rmse": val_rmse, "test_mae": test_mae, "test_rmse": test_rmse}
    }, indent=2), encoding="utf-8")

    return {"val_mae": val_mae, "val_rmse": val_rmse, "test_mae": test_mae, "test_rmse": test_rmse}
