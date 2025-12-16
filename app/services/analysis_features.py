import numpy as np
import pandas as pd

P_YEAR = 365.25

def build_daily_analysis_features(hist: pd.DataFrame) -> pd.DataFrame:
    # expects: date,tmin,tmax,tavg
    df = hist.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # regularize daily
    idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(idx)
    df.index.name = "date"

    # fill small gaps
    for c in ["tmin","tmax","tavg"]:
        df[c] = df[c].astype(float).interpolate(limit_direction="both").ffill().bfill()

    df["diurnal_range"] = df["tmax"] - df["tmin"]
    df["delta_1"] = df["tavg"].diff(1)
    df["delta_7"] = df["tavg"].diff(7)

    df["roll_mean_7"] = df["tavg"].rolling(7).mean()
    df["roll_std_7"] = df["tavg"].rolling(7).std()

    doy = df.index.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / P_YEAR)
    df["doy_cos"] = np.cos(2 * np.pi * doy / P_YEAR)

    t0 = df.index.min()
    df["time_idx"] = (df.index - t0).days.astype(float) / P_YEAR

    # anomaly z-score relative to day-of-year climatology
    tmp = df[["tavg"]].copy()
    tmp["doy"] = df.index.dayofyear
    clim = tmp.groupby("doy")["tavg"].agg(["mean", "std"]).rename(columns={"mean":"mu","std":"sd"})
    df["anomaly_z"] = [
        0.0 if clim.loc[d, "sd"] == 0 or np.isnan(clim.loc[d, "sd"])
        else float((df.loc[dt, "tavg"] - clim.loc[d, "mu"]) / clim.loc[d, "sd"])
        for dt, d in zip(df.index, df.index.dayofyear)
    ]

    out = df.reset_index()
    out["date"] = out["date"].dt.date.astype(str)

    # drop early NaNs from rolling/diff
    out = out.dropna()
    return out

def build_monthly_analysis(daily_feat: pd.DataFrame) -> pd.DataFrame:
    x = daily_feat.copy()
    x["date"] = pd.to_datetime(x["date"])
    x["year"] = x["date"].dt.year.astype(int)
    x["month"] = x["date"].dt.month.astype(int)

    g = x.groupby(["year","month"], as_index=False).agg(
        tavg_mean=("tavg","mean"),
        tavg_std=("tavg","std"),
        diurnal_mean=("diurnal_range","mean"),
        roll_std_mean=("roll_std_7","mean"),
        anomaly_mean=("anomaly_z","mean"),
        delta_1_mean=("delta_1","mean"),
    )
    g = g.fillna(0.0)
    return g
