import numpy as np
import pandas as pd

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _month_name(m: int) -> str:
    return MONTH_NAMES[m-1] if 1 <= m <= 12 else str(m)

def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    # returns slope per 1 unit x
    if len(x) < 2:
        return 0.0
    m, _b = np.polyfit(x.astype(float), y.astype(float), 1)
    return float(m)

def _longest_streak(dates: pd.Series, mask: pd.Series):
    # mask: True/False series aligned with dates
    best_len = 0
    best_start = None
    best_end = None

    cur_len = 0
    cur_start = None

    for d, ok in zip(dates, mask):
        if ok:
            if cur_len == 0:
                cur_start = d
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
                best_end = d
        else:
            cur_len = 0
            cur_start = None

    if best_len == 0:
        return {"length": 0, "start": None, "end": None}
    return {"length": int(best_len), "start": str(best_start.date()), "end": str(best_end.date())}

def _label_cluster(signature_top5: list[dict]):
    # signature items look like: {"feature":"tavg_std","zscore":..., "direction":"high/low"}
    sig = {x["feature"]: x for x in signature_top5}

    def is_high(feat): return feat in sig and sig[feat]["direction"] == "high"
    def is_low(feat): return feat in sig and sig[feat]["direction"] == "low"

    # Heuristics on monthly features we used in triclustering:
    # tavg_mean, tavg_std, diurnal_mean, roll_std_mean, anomaly_mean, delta_1_mean
    if is_low("diurnal_mean") and is_low("tavg_std") and is_low("roll_std_mean"):
        return "Stable cloudy-like regime"
    if is_high("tavg_mean") and is_high("diurnal_mean"):
        return "Dry heat regime"
    if is_high("delta_1_mean") or is_high("roll_std_mean") or is_high("tavg_std"):
        return "Transition / volatile regime"
    if is_low("tavg_mean"):
        return "Cool season regime"
    return "Seasonal regime"

def compute_insights_payload(city_key: str, daily_feat_df: pd.DataFrame, monthly_df: pd.DataFrame, tri: dict, run_id: str):
    # daily_feat_df: columns include date,tavg,tmin,tmax,diurnal_range,roll_std_7,anomaly_z,delta_1,...
    daily = daily_feat_df.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    monthly = monthly_df.copy()
    if not monthly.empty:
        monthly["year"] = monthly["year"].astype(int)
        monthly["month"] = monthly["month"].astype(int)

    data_start = str(daily["date"].min().date()) if len(daily) else None
    data_end = str(daily["date"].max().date()) if len(daily) else None

    # ---- Monthly baseline over all years (12 months) ----
    baseline = []
    if not monthly.empty:
        bm = monthly.groupby("month", as_index=False).agg(
            tavg_mean=("tavg_mean","mean"),
            tavg_std=("tavg_std","mean"),
            diurnal_mean=("diurnal_mean","mean"),
        )
        bm = bm.sort_values("month")
        for _, r in bm.iterrows():
            baseline.append({
                "month": int(r["month"]),
                "month_name": _month_name(int(r["month"])),
                "tavg_mean": float(r["tavg_mean"]),
                "tavg_std": float(r["tavg_std"]) if not pd.isna(r["tavg_std"]) else 0.0,
                "diurnal_mean": float(r["diurnal_mean"]) if not pd.isna(r["diurnal_mean"]) else 0.0,
            })

    hottest_month = None
    coolest_month = None
    if baseline:
        hottest_month = max(baseline, key=lambda x: x["tavg_mean"])
        coolest_month = min(baseline, key=lambda x: x["tavg_mean"])

    # ---- Annual trend (°C/year) based on annual mean of tavg_mean ----
    annual_trend = 0.0
    if not monthly.empty:
        ay = monthly.groupby("year", as_index=False).agg(annual_mean=("tavg_mean","mean"))
        x = ay["year"].values
        y = ay["annual_mean"].values
        annual_trend = _linear_slope(x, y)

    # ---- Extremes & anomalies ----
    top_hot_days = []
    top_cold_days = []
    top_anomaly_days = []
    warm_streak = {"length": 0, "start": None, "end": None}
    cold_streak = {"length": 0, "start": None, "end": None}
    anomaly_count = 0

    if not daily.empty:
        tmp = daily[["date","tavg","anomaly_z"]].copy()
        tmp["abs_anom"] = tmp["anomaly_z"].abs()

        top_hot = daily.sort_values("tavg", ascending=False).head(10)
        top_cold = daily.sort_values("tavg", ascending=True).head(10)
        top_anom = tmp.sort_values("abs_anom", ascending=False).head(10)

        top_hot_days = [{"date": str(d.date()), "tavg": float(v)} for d, v in zip(top_hot["date"], top_hot["tavg"])]
        top_cold_days = [{"date": str(d.date()), "tavg": float(v)} for d, v in zip(top_cold["date"], top_cold["tavg"])]
        top_anomaly_days = [{"date": str(d.date()), "anomaly_z": float(z)} for d, z in zip(top_anom["date"], top_anom["anomaly_z"])]

        anomaly_count = int((daily["anomaly_z"].abs() >= 2.0).sum())

        warm_streak = _longest_streak(daily["date"], daily["anomaly_z"] >= 2.0)
        cold_streak = _longest_streak(daily["date"], daily["anomaly_z"] <= -2.0)

    # ---- Data health ----
    data_health = {}
    if data_start and data_end:
        expected_days = (pd.to_datetime(data_end) - pd.to_datetime(data_start)).days + 1
        actual_days = int(len(daily))
        fill_ratio = 1.0 if expected_days <= 0 else actual_days / expected_days
        data_health = {
            "data_start": data_start,
            "data_end": data_end,
            "expected_days": int(expected_days),
            "actual_days": int(actual_days),
            "coverage_ratio": float(fill_ratio),
        }

    # ---- Cluster labeling ----
    clusters_out = []
    if isinstance(tri, dict):
        for c in tri.get("clusters", []):
            label = _label_cluster(c.get("signature_top5", []))
            months = c.get("months", [])
            clusters_out.append({
                "label": label,
                "years": c.get("years", []),
                "months": months,
                "months_names": [_month_name(int(m)) for m in months],
                "signature_top5": c.get("signature_top5", [])
            })

    payload = {
        "city_key": city_key,
        "analysis_run_id": run_id,
        "data_start": data_start,
        "data_end": data_end,

        "summary": {
            "hottest_month": hottest_month,
            "coolest_month": coolest_month,
            "annual_trend_c_per_year": float(annual_trend),
            "anomaly_days_count_abs_ge_2": int(anomaly_count),
        },

        "monthly_baseline": baseline,

        "extremes": {
            "top_hot_days": top_hot_days,
            "top_cold_days": top_cold_days,
            "top_anomaly_days": top_anomaly_days,
            "warm_anomaly_streak_abs_ge_2": warm_streak,
            "cold_anomaly_streak_abs_ge_2": cold_streak,
        },

        "patterns": {
            "clusters": clusters_out
        },

        "data_health": data_health
    }

    return payload
