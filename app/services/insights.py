import re
import numpy as np
import pandas as pd

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _month_name(m: int) -> str:
    return MONTH_NAMES[m - 1] if 1 <= m <= 12 else str(m)


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    m, _b = np.polyfit(x.astype(float), y.astype(float), 1)
    return float(m)


def _longest_streak(dates: pd.Series, mask: pd.Series):
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
    return {
        "length": int(best_len),
        "start": str(best_start.date()),
        "end": str(best_end.date()),
    }


# -----------------------------------------------------------------------------
# Legacy labeling helpers
# -----------------------------------------------------------------------------

def _label_cluster(signature_top5: list[dict]):
    sig = {x["feature"]: x for x in signature_top5}

    def is_high(feat):
        return feat in sig and sig[feat].get("direction") == "high"

    def is_low(feat):
        return feat in sig and sig[feat].get("direction") == "low"

    if is_low("diurnal_mean") and is_low("tavg_std") and is_low("roll_std_mean"):
        return "Stable cloudy-like regime"
    if is_high("tavg_mean") and is_high("diurnal_mean"):
        return "Dry heat regime"
    if is_high("delta_1_mean") or is_high("roll_std_mean") or is_high("tavg_std"):
        return "Transition / volatile regime"
    if is_low("tavg_mean"):
        return "Cool season regime"
    return "Seasonal regime"


# -----------------------------------------------------------------------------
# TriHSPAM helpers
# -----------------------------------------------------------------------------

_ITEM_RE = re.compile(r"f(\d+)(?:_(\d+))?#(.+)$")


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _parse_pattern_string(pattern_string: str, feature_columns: list[str]) -> list[dict]:
    """
    Parse TriHSPAM pattern string items like:
        "(f0_0#bin1 f7_0#winter) (f0_1#bin2)"
    into structured tokens.
    """
    tokens = re.findall(r"f\d+(?:_\d+)?#[^\s()]+", pattern_string or "")
    out = []

    for tok in tokens:
        m = _ITEM_RE.match(tok)
        if not m:
            continue

        feat_idx = _safe_int(m.group(1))
        ctx_idx = _safe_int(m.group(2), default=None)
        symbol = m.group(3)

        feature_name = f"feature_{feat_idx}"
        if feat_idx is not None and 0 <= feat_idx < len(feature_columns):
            feature_name = feature_columns[feat_idx]

        out.append(
            {
                "raw": tok,
                "feature_idx": feat_idx,
                "context_idx": ctx_idx,
                "symbol": symbol,
                "feature_name": feature_name,
            }
        )

    return out


def _feature_signature_from_tokens(tokens: list[dict]) -> list[dict]:
    """
    Convert parsed TriHSPAM tokens into a dashboard-friendly signature_top5 list.
    Format kept compatible with old frontend:
        [{"feature": "...", "direction": "..."}]
    """
    if not tokens:
        return []

    counts = {}
    for t in tokens:
        key = (t["feature_name"], t["symbol"])
        counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[:5]

    sig = []
    for (feature_name, symbol), cnt in ranked:
        sig.append(
            {
                "feature": feature_name,
                "direction": str(symbol),
                "count": int(cnt),
            }
        )
    return sig


def _months_years_from_windows(row_windows_meta: list[dict]) -> tuple[list[int], list[str], list[int]]:
    months = []
    years = []

    for w in row_windows_meta or []:
        center = w.get("center_date") or w.get("start_date") or w.get("end_date")
        if not center:
            continue
        dt = pd.to_datetime(center, errors="coerce")
        if pd.isna(dt):
            continue
        months.append(int(dt.month))
        years.append(int(dt.year))

    uniq_months = sorted(set(months))
    uniq_years = sorted(set(years))
    month_names = [_month_name(m) for m in uniq_months]
    return uniq_months, month_names, uniq_years


def _dominant_values(tokens: list[dict], feature_name: str) -> list[str]:
    vals = [t["symbol"] for t in tokens if t["feature_name"] == feature_name]
    if not vals:
        return []
    s = pd.Series(vals, dtype="object")
    vc = s.value_counts()
    return vc.index.tolist()


def _label_trihspam_regime(tric: dict, feature_columns: list[str]) -> str:
    """
    Heuristic labeler for TriHSPAM triclusters.

    We use:
    - symbolic tokens in the mined pattern
    - included features
    - support / shape
    - season tokens if present
    """
    tokens = _parse_pattern_string(tric.get("pattern_string", ""), feature_columns)
    feature_names = set(tric.get("feature_names", []))

    seasons = _dominant_values(tokens, "season_class")
    temp_bands = _dominant_values(tokens, "temp_band")
    trend_vals = _dominant_values(tokens, "trend_class")
    anom_vals = _dominant_values(tokens, "anomaly_class")
    diurnal_vals = _dominant_values(tokens, "diurnal_band")

    if "season_class" in feature_names:
        if "monsoon" in seasons:
            return "Monsoon regime"
        if "winter" in seasons:
            return "Winter regime"
        if "pre_monsoon" in seasons:
            return "Pre-monsoon regime"
        if "post_monsoon" in seasons:
            return "Post-monsoon regime"

    if "temp_band" in feature_names and temp_bands:
        if temp_bands[0] == "hot":
            if "trend_class" in feature_names and "warming" in trend_vals:
                return "Hot warming regime"
            return "Hot regime"
        if temp_bands[0] == "cold":
            if "anomaly_class" in feature_names and "low" in anom_vals:
                return "Cold anomaly regime"
            return "Cool regime"
        if temp_bands[0] == "mild" and "trend_class" in feature_names and "stable" in trend_vals:
            return "Stable mild regime"

    if "anomaly_class" in feature_names and anom_vals:
        if anom_vals[0] == "high":
            return "Warm anomaly regime"
        if anom_vals[0] == "low":
            return "Cold anomaly regime"

    if "diurnal_band" in feature_names and diurnal_vals:
        if diurnal_vals[0] == "high":
            return "High diurnal variability regime"
        if diurnal_vals[0] == "low":
            return "Low diurnal variability regime"

    if "trend_class" in feature_names and trend_vals:
        if trend_vals[0] == "warming":
            return "Warming transition regime"
        if trend_vals[0] == "cooling":
            return "Cooling transition regime"
        if trend_vals[0] == "stable":
            return "Stable regime"

    if tric.get("type") == "mixed":
        return "Mixed weather regime"
    if tric.get("type") == "symbolic":
        return "Symbolic seasonal regime"
    return "Numeric weather regime"


def _serialize_tricluster(tric: dict, feature_columns: list[str]) -> dict:
    tokens = _parse_pattern_string(tric.get("pattern_string", ""), feature_columns)
    months, month_names, years = _months_years_from_windows(tric.get("row_windows_meta", []))
    signature_top5 = _feature_signature_from_tokens(tokens)
    label = _label_trihspam_regime(tric, feature_columns)

    return {
        "id": tric.get("id"),
        "label": label,
        "type": tric.get("type"),
        "support": int(tric.get("support", 0)),
        "hvar3": _safe_float(tric.get("hvar3", 0.0)),
        "rows": tric.get("rows", []),
        "cols": tric.get("cols", []),
        "contexts": tric.get("contexts", []),
        "feature_names": tric.get("feature_names", []),
        "feature_groups": tric.get("feature_groups", {"numeric": [], "symbolic": []}),
        "shape": tric.get("shape", {}),
        "months": months,
        "months_names": month_names,
        "years": years,
        "signature_top5": signature_top5,
        "row_window_ids": tric.get("row_window_ids", []),
        "row_windows_meta": tric.get("row_windows_meta", []),
        "pattern_string": tric.get("pattern_string"),
    }


def _build_pattern_summary_from_trihspam(tri: dict) -> dict:
    feature_columns = tri.get("feature_columns", [])
    raw_trics = tri.get("triclusters", []) if isinstance(tri, dict) else []

    triclusters = [_serialize_tricluster(t, feature_columns) for t in raw_trics]

    triclusters_sorted = sorted(
        triclusters,
        key=lambda x: (
            x.get("hvar3", 9999.0),
            -x.get("support", 0),
            -(x.get("shape", {}).get("volume", 0)),
        ),
    )

    dashboard_clusters = []
    for t in triclusters_sorted[:12]:
        dashboard_clusters.append(
            {
                "label": t["label"],
                "years": t["years"],
                "months": t["months"],
                "months_names": t["months_names"],
                "signature_top5": t["signature_top5"],
                "support": t["support"],
                "hvar3": t["hvar3"],
                "type": t["type"],
            }
        )

    total_support_windows = set()
    for t in triclusters:
        for wid in t.get("row_window_ids", []):
            total_support_windows.add(wid)

    top_coherent = triclusters_sorted[:5]
    most_supported = sorted(triclusters, key=lambda x: (-x["support"], x["hvar3"]))[:5]
    largest_volume = sorted(
        triclusters,
        key=lambda x: (-(x.get("shape", {}).get("volume", 0)), x["hvar3"])
    )[:5]

    return {
        "method": tri.get("method", "TriHSPAM"),
        "engine": tri.get("engine", {}),
        "config": tri.get("config", {}),
        "clusters": dashboard_clusters,      # kept for current dashboard compatibility
        "triclusters": triclusters_sorted,   # richer structure for future UI
        "summary": {
            "n_triclusters": int(len(triclusters)),
            "n_supported_windows": int(len(total_support_windows)),
            "top_coherent_ids": [x["id"] for x in top_coherent],
            "most_supported_ids": [x["id"] for x in most_supported],
            "largest_volume_ids": [x["id"] for x in largest_volume],
        },
    }


def _build_pattern_summary_legacy(tri: dict) -> dict:
    clusters_out = []
    for c in tri.get("clusters", []):
        label = _label_cluster(c.get("signature_top5", []))
        months = c.get("months", [])
        clusters_out.append(
            {
                "label": label,
                "years": c.get("years", []),
                "months": months,
                "months_names": [_month_name(int(m)) for m in months],
                "signature_top5": c.get("signature_top5", []),
            }
        )

    return {
        "method": tri.get("method", "legacy_kmeans_placeholder"),
        "clusters": clusters_out,
        "triclusters": [],
        "summary": {
            "n_triclusters": 0,
            "n_supported_windows": 0,
            "top_coherent_ids": [],
            "most_supported_ids": [],
            "largest_volume_ids": [],
        },
    }


# -----------------------------------------------------------------------------
# Main insights payload
# -----------------------------------------------------------------------------

def compute_insights_payload(
    city_key: str,
    daily_feat_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    tri: dict,
    run_id: str,
):
    daily = daily_feat_df.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    monthly = monthly_df.copy()
    if not monthly.empty:
        monthly["year"] = monthly["year"].astype(int)
        monthly["month"] = monthly["month"].astype(int)

    data_start = str(daily["date"].min().date()) if len(daily) else None
    data_end = str(daily["date"].max().date()) if len(daily) else None

    # -------------------------------------------------------------------------
    # Monthly climate baseline
    # -------------------------------------------------------------------------
    baseline = []
    if not monthly.empty:
        bm = monthly.groupby("month", as_index=False).agg(
            tavg_mean=("tavg_mean", "mean"),
            tavg_std=("tavg_std", "mean"),
            diurnal_mean=("diurnal_mean", "mean"),
        )
        bm = bm.sort_values("month")

        for _, r in bm.iterrows():
            baseline.append(
                {
                    "month": int(r["month"]),
                    "month_name": _month_name(int(r["month"])),
                    "tavg_mean": float(r["tavg_mean"]),
                    "tavg_std": float(r["tavg_std"]) if not pd.isna(r["tavg_std"]) else 0.0,
                    "diurnal_mean": float(r["diurnal_mean"]) if not pd.isna(r["diurnal_mean"]) else 0.0,
                }
            )

    hottest_month = None
    coolest_month = None
    if baseline:
        hottest_month = max(baseline, key=lambda x: x["tavg_mean"])
        coolest_month = min(baseline, key=lambda x: x["tavg_mean"])

    # -------------------------------------------------------------------------
    # Annual trend
    # -------------------------------------------------------------------------
    annual_trend = 0.0
    if not monthly.empty:
        ay = monthly.groupby("year", as_index=False).agg(annual_mean=("tavg_mean", "mean"))
        x = ay["year"].values
        y = ay["annual_mean"].values
        annual_trend = _linear_slope(x, y)

    # -------------------------------------------------------------------------
    # Extremes / anomalies
    # -------------------------------------------------------------------------
    top_hot_days = []
    top_cold_days = []
    top_anomaly_days = []
    warm_streak = {"length": 0, "start": None, "end": None}
    cold_streak = {"length": 0, "start": None, "end": None}
    anomaly_count = 0

    if not daily.empty:
        tmp = daily[["date", "tavg", "anomaly_z"]].copy()
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

    # -------------------------------------------------------------------------
    # Data health
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Patterns: support both legacy and TriHSPAM
    # -------------------------------------------------------------------------
    tri = tri or {}
    tri_method = str(tri.get("method", "")).lower()

    if tri_method == "trihspam" or "triclusters" in tri:
        patterns = _build_pattern_summary_from_trihspam(tri)
    else:
        patterns = _build_pattern_summary_legacy(tri)

    # -------------------------------------------------------------------------
    # Summary cards
    # -------------------------------------------------------------------------
    summary = {
        "hottest_month": hottest_month,
        "coolest_month": coolest_month,
        "annual_trend_c_per_year": float(annual_trend),
        "anomaly_days_count_abs_ge_2": int(anomaly_count),
        "n_patterns": int(patterns.get("summary", {}).get("n_triclusters", 0)),
    }

    payload = {
        "city_key": city_key,
        "analysis_run_id": run_id,
        "data_start": data_start,
        "data_end": data_end,
        "summary": summary,
        "monthly_baseline": baseline,
        "extremes": {
            "top_hot_days": top_hot_days,
            "top_cold_days": top_cold_days,
            "top_anomaly_days": top_anomaly_days,
            "warm_anomaly_streak_abs_ge_2": warm_streak,
            "cold_anomaly_streak_abs_ge_2": cold_streak,
        },
        "patterns": patterns,
        "data_health": data_health,
    }

    return payload