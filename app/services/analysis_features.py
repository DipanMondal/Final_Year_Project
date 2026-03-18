import numpy as np
import pandas as pd

P_YEAR = 365.25

NUMERIC_FEATURES_V1 = [
    "tavg",
    "tmin",
    "tmax",
    "diurnal_range",
    "delta_1",
    "roll_std_7",
    "anomaly_z",
]

SYMBOLIC_FEATURES_V1 = [
    "temp_band",
    "trend_class",
    "anomaly_class",
    "diurnal_band",
    "season_class",
]

FEATURE_COLUMNS_V1 = NUMERIC_FEATURES_V1 + SYMBOLIC_FEATURES_V1


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
    for c in ["tmin", "tmax", "tavg"]:
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
    clim = tmp.groupby("doy")["tavg"].agg(["mean", "std"]).rename(columns={"mean": "mu", "std": "sd"})
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

    g = x.groupby(["year", "month"], as_index=False).agg(
        tavg_mean=("tavg", "mean"),
        tavg_std=("tavg", "std"),
        diurnal_mean=("diurnal_range", "mean"),
        roll_std_mean=("roll_std_7", "mean"),
        anomaly_mean=("anomaly_z", "mean"),
        delta_1_mean=("delta_1", "mean"),
    )
    g = g.fillna(0.0)
    return g


# -----------------------------------------------------------------------------
# TriHSPAM preparation helpers
# -----------------------------------------------------------------------------

def _validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_quantile_labels(
    series: pd.Series,
    low_label: str,
    mid_label: str,
    high_label: str,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    q1 = float(s.quantile(1.0 / 3.0))
    q2 = float(s.quantile(2.0 / 3.0))

    def label(v: float) -> str:
        if pd.isna(v):
            return "missing"
        if v <= q1:
            return low_label
        if v <= q2:
            return mid_label
        return high_label

    return s.apply(label)


def _map_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "pre_monsoon"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


def build_enriched_daily_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Build a regularized daily heterogeneous feature table for TriHSPAM.

    Expected input columns:
        - date
        - tmin
        - tmax
        - tavg

    Returns a dataframe with both numeric and symbolic features.
    """
    _validate_required_columns(hist, ["date", "tmin", "tmax", "tavg"])

    df = hist.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    if df.empty:
        raise ValueError("No valid dated weather history rows were provided.")

    # Regularize to daily frequency
    idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(idx)
    df.index.name = "date"

    # Numeric cleanup
    for c in ["tmin", "tmax", "tavg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].interpolate(limit_direction="both").ffill().bfill()

    if df[["tmin", "tmax", "tavg"]].isna().any().any():
        raise ValueError("Unable to regularize numeric weather columns after interpolation.")

    # Derived numeric features
    df["diurnal_range"] = df["tmax"] - df["tmin"]
    df["delta_1"] = df["tavg"].diff(1)
    df["delta_7"] = df["tavg"].diff(7)
    df["roll_mean_7"] = df["tavg"].rolling(7, min_periods=7).mean()
    df["roll_std_7"] = df["tavg"].rolling(7, min_periods=7).std()

    doy = df.index.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / P_YEAR)
    df["doy_cos"] = np.cos(2 * np.pi * doy / P_YEAR)

    t0 = df.index.min()
    df["time_idx"] = (df.index - t0).days.astype(float) / P_YEAR

    # anomaly z-score relative to day-of-year climatology
    tmp = pd.DataFrame({"tavg": df["tavg"], "doy": df.index.dayofyear}, index=df.index)
    clim = tmp.groupby("doy")["tavg"].agg(["mean", "std"]).rename(columns={"mean": "mu", "std": "sd"})

    anomaly_vals = []
    for dt, d in zip(df.index, df.index.dayofyear):
        mu = clim.loc[d, "mu"]
        sd = clim.loc[d, "sd"]
        if pd.isna(sd) or sd == 0:
            anomaly_vals.append(0.0)
        else:
            anomaly_vals.append(float((df.loc[dt, "tavg"] - mu) / sd))
    df["anomaly_z"] = anomaly_vals

    # Fill early rolling/diff NaNs after deriving stable features
    df["delta_1"] = df["delta_1"].fillna(0.0)
    df["delta_7"] = df["delta_7"].fillna(0.0)
    df["roll_mean_7"] = df["roll_mean_7"].bfill().ffill()
    df["roll_std_7"] = df["roll_std_7"].bfill().ffill().fillna(0.0)
    df["diurnal_range"] = df["diurnal_range"].fillna(0.0)
    df["anomaly_z"] = df["anomaly_z"].fillna(0.0)

    # Symbolic features for heterogeneous TriHSPAM input
    df["temp_band"] = _safe_quantile_labels(df["tavg"], "cold", "mild", "hot")
    df["trend_class"] = np.where(
        df["delta_1"] < -1.0,
        "cooling",
        np.where(df["delta_1"] > 1.0, "warming", "stable"),
    )
    df["anomaly_class"] = np.where(
        df["anomaly_z"] < -1.0,
        "low",
        np.where(df["anomaly_z"] > 1.0, "high", "normal"),
    )
    df["diurnal_band"] = _safe_quantile_labels(df["diurnal_range"], "low", "medium", "high")
    df["season_class"] = pd.Series(df.index.month, index=df.index).apply(_map_season)

    out = df.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    ordered_cols = [
        "date",
        "tmin",
        "tmax",
        "tavg",
        "diurnal_range",
        "delta_1",
        "delta_7",
        "roll_mean_7",
        "roll_std_7",
        "doy_sin",
        "doy_cos",
        "time_idx",
        "anomaly_z",
        "temp_band",
        "trend_class",
        "anomaly_class",
        "diurnal_band",
        "season_class",
    ]
    available_cols = [c for c in ordered_cols if c in out.columns]
    return out[available_cols].copy()


def build_weather_windows(
    daily_df: pd.DataFrame,
    window_size: int = 30,
    stride: int = 7,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Build rolling weather windows from an enriched daily dataframe.

    Returns:
        windows_meta: list of metadata dicts
        windows_df: long-form dataframe with one row per day within each window
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    _validate_required_columns(daily_df, ["date"])

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if len(df) < window_size:
        raise ValueError(
            f"Not enough rows to create one weather window: have {len(df)}, need at least {window_size}."
        )

    windows_meta: list[dict] = []
    window_frames: list[pd.DataFrame] = []

    for start_idx in range(0, len(df) - window_size + 1, stride):
        end_idx = start_idx + window_size
        window_id = len(windows_meta)
        window_slice = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        center_idx = int(window_size // 2)

        windows_meta.append(
            {
                "window_id": window_id,
                "start_date": window_slice.loc[0, "date"].strftime("%Y-%m-%d"),
                "end_date": window_slice.loc[window_size - 1, "date"].strftime("%Y-%m-%d"),
                "center_date": window_slice.loc[center_idx, "date"].strftime("%Y-%m-%d"),
            }
        )

        window_slice["window_id"] = window_id
        window_slice["context_idx"] = np.arange(window_size, dtype=int)
        window_frames.append(window_slice)

    if not window_frames:
        raise ValueError("No weather windows were generated from the provided dataframe.")

    windows_df = pd.concat(window_frames, ignore_index=True)
    windows_df["date"] = windows_df["date"].dt.strftime("%Y-%m-%d")

    first_cols = ["window_id", "context_idx", "date"]
    remaining_cols = [c for c in windows_df.columns if c not in first_cols]
    windows_df = windows_df[first_cols + remaining_cols]
    return windows_meta, windows_df


def build_trihspam_cube(
    windows_df: pd.DataFrame,
    feature_columns: list[str],
    numeric_features: list[str],
    symbolic_features: list[str],
    window_size: int,
) -> dict:
    """
    Convert long-form weather windows into a TriHSPAM-ready cube.

    Returned cube shape:
        (n_features, n_windows, window_size)

    Mixed-type data is stored using object dtype.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    required = ["window_id", "context_idx", "date"] + list(feature_columns)
    _validate_required_columns(windows_df, required)

    invalid_numeric = [c for c in numeric_features if c not in feature_columns]
    invalid_symbolic = [c for c in symbolic_features if c not in feature_columns]
    if invalid_numeric:
        raise ValueError(f"Numeric features not present in feature_columns: {invalid_numeric}")
    if invalid_symbolic:
        raise ValueError(f"Symbolic features not present in feature_columns: {invalid_symbolic}")

    df = windows_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["window_id", "context_idx"]).reset_index(drop=True)

    window_ids = sorted(df["window_id"].drop_duplicates().tolist())
    if not window_ids:
        raise ValueError("No windows found in windows_df.")

    counts = df.groupby("window_id")["context_idx"].nunique().to_dict()
    bad_windows = [wid for wid, n in counts.items() if n != window_size]
    if bad_windows:
        raise ValueError(
            f"Each window must contain exactly {window_size} context positions. Bad windows: {bad_windows[:10]}"
        )

    n_features = len(feature_columns)
    n_windows = len(window_ids)
    cube = np.empty((n_features, n_windows, window_size), dtype=object)

    window_id_to_idx = {wid: idx for idx, wid in enumerate(window_ids)}

    for wid in window_ids:
        w_idx = window_id_to_idx[wid]
        chunk = df[df["window_id"] == wid].sort_values("context_idx")
        contexts = chunk["context_idx"].to_numpy()
        expected_contexts = np.arange(window_size)
        if len(contexts) != window_size or not np.array_equal(contexts, expected_contexts):
            raise ValueError(f"Window {wid} has invalid context indices; expected 0..{window_size - 1}.")

        for f_idx, feature in enumerate(feature_columns):
            values = chunk[feature]
            if feature in numeric_features:
                numeric_vals = pd.to_numeric(values, errors="coerce").astype(float).to_numpy()
                cube[f_idx, w_idx, :] = numeric_vals
            else:
                symbolic_vals = values.fillna("missing").astype(str).to_numpy()
                cube[f_idx, w_idx, :] = symbolic_vals

    numeric_feature_indices = [feature_columns.index(f) for f in numeric_features]
    symbolic_feature_indices = [feature_columns.index(f) for f in symbolic_features]

    meta_df = (
        df.groupby("window_id", as_index=False)
        .agg(start_date=("date", "min"), end_date=("date", "max"))
        .sort_values("window_id")
        .reset_index(drop=True)
    )
    meta_df["center_date"] = meta_df.apply(
        lambda r: r["start_date"] + (r["end_date"] - r["start_date"]) / 2,
        axis=1,
    )

    windows_meta = [
        {
            "window_id": int(row.window_id),
            "start_date": pd.Timestamp(row.start_date).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(row.end_date).strftime("%Y-%m-%d"),
            "center_date": pd.Timestamp(row.center_date).strftime("%Y-%m-%d"),
        }
        for row in meta_df.itertuples(index=False)
    ]

    return {
        "cube": cube,
        "feature_columns": list(feature_columns),
        "numeric_features": list(numeric_features),
        "symbolic_features": list(symbolic_features),
        "numeric_feature_indices": numeric_feature_indices,
        "symbolic_feature_indices": symbolic_feature_indices,
        "windows_meta": windows_meta,
        "window_ids": window_ids,
        "n_windows": n_windows,
        "window_size": window_size,
    }