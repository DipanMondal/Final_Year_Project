from pathlib import Path
import json
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

P_YEAR = 365.25

def _regularize_daily(y_raw: pd.Series) -> pd.Series:
    y_raw = y_raw.sort_index()
    full_idx = pd.date_range(y_raw.index.min(), y_raw.index.max(), freq="D")
    y = y_raw.reindex(full_idx)
    y = y.interpolate(limit_direction="both").ffill().bfill()
    return y.astype(float)

def _make_exog(idx: pd.DatetimeIndex, t0: pd.Timestamp, K: int = 3) -> pd.DataFrame:
    doy = idx.dayofyear.values.astype(float)
    out = {}

    for k in range(1, K + 1):
        out[f"sin{k}"] = np.sin(2 * np.pi * k * doy / P_YEAR)
        out[f"cos{k}"] = np.cos(2 * np.pi * k * doy / P_YEAR)

    out["time_idx"] = (idx - t0).days.astype(float) / P_YEAR
    return pd.DataFrame(out, index=idx)

def _rolling_cv_mae(y: pd.Series, exog: pd.DataFrame, order, seasonal_order, folds=3, horizon=30):
    n = len(y)
    needed = folds * horizon + 30
    if n < needed:
        folds = 2
    if n < 2 * horizon + 30:
        folds = 1

    maes = []
    for i in range(folds):
        val_end = n - (folds - i - 1) * horizon
        val_start = val_end - horizon

        y_tr = y.iloc[:val_start]
        x_tr = exog.iloc[:val_start]
        y_va = y.iloc[val_start:val_end]
        x_va = exog.iloc[val_start:val_end]

        model = SARIMAX(
            y_tr,
            exog=x_tr,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False, maxiter=200)

        # if optimizer didn't converge, skip
        if hasattr(res, "mle_retvals") and not res.mle_retvals.get("converged", True):
            return None

        fc = res.get_forecast(steps=len(y_va), exog=x_va).predicted_mean
        maes.append(mean_absolute_error(y_va.values, fc.values))

    return float(np.mean(maes)) if maes else None

def train_city_sarimax(city_key: str, y_raw: pd.Series, fourier_K: int = 3):
    y = _regularize_daily(y_raw)
    if len(y) < 3 * 365:
        raise ValueError("Not enough daily data. Use at least ~3 years.")

    t0 = y.index.min()
    exog = _make_exog(y.index, t0=t0, K=fourier_K)

    # Keep grid reasonable (fast + stable). You can expand later.
    candidate_orders = [
        (1,1,1), (2,1,1), (1,1,2), (2,1,2),
        (1,0,1), (0,1,1)
    ]
    candidate_seasonals = [
        (1,0,1,7),
        (1,1,1,7),
        (0,1,1,7),
        (1,0,0,7),
    ]

    best = None
    best_score = float("inf")

    for order in candidate_orders:
        for seas in candidate_seasonals:
            try:
                score = _rolling_cv_mae(y, exog, order, seas, folds=3, horizon=30)
                if score is None:
                    continue
                if score < best_score:
                    best_score = score
                    best = (order, seas)
            except Exception:
                continue

    if best is None:
        raise ValueError("Could not fit any SARIMAX model. Try fewer years or smaller grid.")

    best_order, best_seasonal = best

    # Fit final on full series
    final_model = SARIMAX(
        y,
        exog=exog,
        order=best_order,
        seasonal_order=best_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    final_res = final_model.fit(disp=False, maxiter=300)

    out_dir = Path("artifacts/models") / city_key
    out_dir.mkdir(parents=True, exist_ok=True)

    final_res.save(str(out_dir / "sarimax.pkl"))

    meta = {
        "city": city_key,
        "order": best_order,
        "seasonal_order": best_seasonal,
        "seasonal_period": 7,
        "fourier_K": fourier_K,
        "cv_folds": 3,
        "cv_horizon_days": 30,
        "cv_mae": float(best_score),
        "train_start": str(y.index.min().date()),
        "train_end": str(y.index.max().date()),
        "exog_cols": list(exog.columns),
        "t0": str(t0.date())
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta
