import json
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path
from .db import fetch_history

MODEL_PATH = Path("artifacts/temp_model.joblib")
FEAT_PATH = Path("artifacts/feature_config.json")
META_PATH = Path("artifacts/train_metadata.json")

class TempForecaster:
    def __init__(self):
        self.model = load(MODEL_PATH)
        self.feature_cols = json.loads(FEAT_PATH.read_text(encoding="utf-8"))["feature_cols"]
        self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    def _make_row_features(self, dates, tavg_series):
        # dates: pd.Timestamp, tavg_series: pd.Series of historical+pred
        df = pd.DataFrame({"date": dates, "tavg": tavg_series})
        df["tavg_lag_1"] = df["tavg"].shift(1)
        df["tavg_lag_2"] = df["tavg"].shift(2)
        df["tavg_lag_7"] = df["tavg"].shift(7)
        df["tavg_roll_mean_7"] = df["tavg"].shift(1).rolling(7).mean()
        df["tavg_roll_std_7"] = df["tavg"].shift(1).rolling(7).std()

        doy = df["date"].dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        return df

    def forecast(self, city: str, horizon_days: int = 7):
        hist = fetch_history(city, None, None)
        if hist.empty:
            raise ValueError("No data for this city. Ingest & train first.")

        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values("date")
        series = hist[["date","tavg"]].dropna().copy()

        # Need at least 8 days for lag/rolling features
        if len(series) < 15:
            raise ValueError("Not enough history. Need at least ~15 days.")

        # recursive multi-step forecast
        dates = series["date"].tolist()
        temps = series["tavg"].astype(float).tolist()

        preds = []
        last_date = dates[-1]

        for i in range(horizon_days):
            next_date = last_date + pd.Timedelta(days=1)

            dates.append(next_date)
            temps.append(np.nan)  # placeholder

            df_feat = self._make_row_features(pd.Series(dates), pd.Series(temps))
            row = df_feat.iloc[-1][self.feature_cols].to_frame().T

            # if row has NaN due to missing lags, stop
            if row.isna().any(axis=1).iloc[0]:
                raise ValueError("Feature build failed due to insufficient clean history.")

            yhat = float(self.model.predict(row)[0])

            temps[-1] = yhat
            preds.append({"date": next_date.date().isoformat(), "tavg": yhat})
            last_date = next_date

        return {
            "city": city,
            "horizon_days": horizon_days,
            "predictions": preds,
            "model_info": self.meta
        }
