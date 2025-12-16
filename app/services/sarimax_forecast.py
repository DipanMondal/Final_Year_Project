from pathlib import Path
import json
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from .sarimax_train import _make_exog, P_YEAR
from .db import fetch_history

class CitySarimaxForecaster:
    def __init__(self, city_key: str):
        self.city_key = city_key
        base = Path("artifacts/models") / city_key
        self.model_path = base / "sarimax.pkl"
        self.meta_path = base / "meta.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"No model for {city_key}. Train first via POST /cities.")

        self.res = SARIMAXResults.load(str(self.model_path))
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def forecast(self, horizon_days: int = 7):
        # create future exog using same t0 used in training
        fourier_K = int(self.meta.get("fourier_K", 3))
        t0 = pd.to_datetime(self.meta.get("t0", self.meta["train_start"]))

        hist = fetch_history(self.city_key, None, None)
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values("date")
        last_date = hist["date"].max()

        future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        ex_future = _make_exog(future_idx, t0=t0, K=fourier_K)

        fc = self.res.get_forecast(steps=horizon_days, exog=ex_future)
        mean = fc.predicted_mean

        ci = fc.conf_int(alpha=0.05)
        ci.columns = ["lower", "upper"]

        preds = []
        for d in future_idx:
            preds.append({
                "date": d.date().isoformat(),
                "tavg": float(mean.loc[d]),
                "lower_95": float(ci.loc[d, "lower"]),
                "upper_95": float(ci.loc[d, "upper"]),
            })

        return {"city": self.city_key, "horizon_days": horizon_days, "predictions": preds, "model_info": self.meta}

class ModelRegistry:
    def __init__(self):
        self._cache = {}

    def get(self, city_key: str):
        if city_key not in self._cache:
            self._cache[city_key] = CitySarimaxForecaster(city_key)
        return self._cache[city_key]

    def invalidate(self, city_key: str):
        self._cache.pop(city_key, None)
