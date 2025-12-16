from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load
from .db import fetch_history

def city_key(city: str, country_code: str | None = None):
    c = city.strip().lower().replace(" ", "_")
    cc = (country_code or "").strip().lower()
    return f"{c}_{cc}" if cc else c

class CityForecaster:
    def __init__(self, city_key: str):
        self.city_key = city_key
        base = Path("artifacts/models") / city_key
        self.model = load(base / "temp_model.joblib")
        self.feature_cols = json.loads((base / "feature_config.json").read_text(encoding="utf-8"))["feature_cols"]
        self.meta = json.loads((base / "train_metadata.json").read_text(encoding="utf-8"))

    def _make_features(self, dates, temps):
        df = pd.DataFrame({"date": dates, "tavg": temps})
        df["tavg_lag_1"] = df["tavg"].shift(1)
        df["tavg_lag_2"] = df["tavg"].shift(2)
        df["tavg_lag_7"] = df["tavg"].shift(7)
        df["tavg_roll_mean_7"] = df["tavg"].shift(1).rolling(7).mean()
        df["tavg_roll_std_7"] = df["tavg"].shift(1).rolling(7).std()
        doy = pd.to_datetime(df["date"]).dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        return df

    def forecast(self, horizon_days: int):
        hist = fetch_history(self.city_key, None, None)
        if hist.empty:
            raise ValueError("No DB history for this city.")

        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values("date")
        series = hist[["date","tavg"]].dropna()

        if len(series) < 15:
            raise ValueError("Not enough history to forecast.")

        dates = series["date"].tolist()
        temps = series["tavg"].astype(float).tolist()

        preds = []
        last_date = dates[-1]

        for _ in range(horizon_days):
            next_date = last_date + pd.Timedelta(days=1)
            dates.append(next_date)
            temps.append(np.nan)

            df_feat = self._make_features(pd.Series(dates), pd.Series(temps))
            row = df_feat.iloc[-1][self.feature_cols].to_frame().T
            if row.isna().any(axis=1).iloc[0]:
                raise ValueError("Feature build failed. History may be too sparse.")

            yhat = float(self.model.predict(row)[0])
            temps[-1] = yhat
            preds.append({"date": next_date.date().isoformat(), "tavg": yhat})
            last_date = next_date

        return {"city": self.city_key, "horizon_days": horizon_days, "predictions": preds, "model_info": self.meta}

class ModelRegistry:
    def __init__(self):
        self._cache = {}

    def get(self, city_key: str):
        if city_key not in self._cache:
            self._cache[city_key] = CityForecaster(city_key)
        return self._cache[city_key]
