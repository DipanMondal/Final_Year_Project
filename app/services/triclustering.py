import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .analysis_features import (
    FEATURE_COLUMNS_V1,
    NUMERIC_FEATURES_V1,
    SYMBOLIC_FEATURES_V1,
    build_enriched_daily_features,
    build_weather_windows,
    build_trihspam_cube,
)
from .trihspam_engine import TriHSPAMConfig, run_weather_trihspam


# -----------------------------------------------------------------------------
# Legacy monthly placeholder triclustering
# Keep this for fallback while we migrate the pipeline.
# -----------------------------------------------------------------------------

LEGACY_FEATURES = [
    "tavg_mean",
    "tavg_std",
    "diurnal_mean",
    "roll_std_mean",
    "anomaly_mean",
    "delta_1_mean",
]


def tricluster_year_month_features(monthly: pd.DataFrame, k_years: int = 3, k_months: int = 3):
    """
    Legacy placeholder method kept temporarily for compatibility.
    This is the old year-month-feature KMeans approximation.
    """
    if monthly.empty:
        raise ValueError("No monthly analysis data available for triclustering.")

    years = sorted(monthly["year"].unique().tolist())
    Y = len(years)
    F = len(LEGACY_FEATURES)

    T = np.full((Y, 12, F), np.nan, dtype=float)
    year_to_i = {y: i for i, y in enumerate(years)}

    for _, r in monthly.iterrows():
        yi = year_to_i[int(r["year"])]
        mi = int(r["month"]) - 1
        T[yi, mi, :] = [float(r[f]) for f in LEGACY_FEATURES]

    flat = T.reshape(-1, F)
    col_means = np.nanmean(flat, axis=0)
    inds = np.where(np.isnan(flat))
    flat[inds] = np.take(col_means, inds[1])
    T = flat.reshape(Y, 12, F)

    flat2 = T.reshape(-1, F)
    mu = flat2.mean(axis=0)
    sd = flat2.std(axis=0)
    sd[sd == 0] = 1.0
    flat2 = (flat2 - mu) / sd
    Z = flat2.reshape(Y, 12, F)

    ky = min(k_years, Y) if Y >= 2 else 1
    X_year = Z.reshape(Y, 12 * F)

    if ky > 1:
        year_labels = KMeans(n_clusters=ky, n_init=10, random_state=42).fit_predict(X_year)
    else:
        year_labels = np.zeros(Y, dtype=int)

    clusters = []
    for yc in range(ky):
        y_idx = np.where(year_labels == yc)[0]
        years_in = [years[i] for i in y_idx.tolist()]

        M = Z[y_idx].mean(axis=0)  # (12, F)

        km = min(k_months, 12) if 12 >= 2 else 1
        if km > 1:
            month_labels = KMeans(n_clusters=km, n_init=10, random_state=42).fit_predict(M)
        else:
            month_labels = np.zeros(12, dtype=int)

        for mc in range(km):
            m_idx = np.where(month_labels == mc)[0]
            months_in = [int(i + 1) for i in m_idx.tolist()]

            sig = M[m_idx].mean(axis=0)
            order = np.argsort(np.abs(sig))[::-1][:5]
            signature = []
            for j in order:
                signature.append(
                    {
                        "feature": LEGACY_FEATURES[j],
                        "zscore": float(sig[j]),
                        "direction": "high" if sig[j] >= 0 else "low",
                    }
                )

            clusters.append(
                {
                    "years": years_in,
                    "months": months_in,
                    "signature_top5": signature,
                }
            )

    return {
        "method": "legacy_kmeans_placeholder",
        "features_used": LEGACY_FEATURES,
        "k_years": int(ky),
        "k_months": int(k_months),
        "clusters": clusters,
    }


# -----------------------------------------------------------------------------
# New TriHSPAM-based weather triclustering
# -----------------------------------------------------------------------------

DEFAULT_WINDOW_SIZE = 30
DEFAULT_STRIDE = 7
DEFAULT_MIN_I = 3
DEFAULT_MIN_J = 2
DEFAULT_MIN_K = 3
DEFAULT_N_BINS = 5
DEFAULT_DISC_METHOD = "eq_size"
DEFAULT_MV_METHOD = None
DEFAULT_SPM_ALGO = "fournier08closed"
DEFAULT_TIME_RELAXED = False
DEFAULT_COHERENCE_THRESHOLD = 0.5
DEFAULT_OVERLAP_FILTER = 0.8


def _clean_positive_int(value, name: str) -> int:
    iv = int(value)
    if iv <= 0:
        raise ValueError(f"{name} must be positive.")
    return iv


def run_weather_triclustering_from_history(
    hist_df: pd.DataFrame,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    min_I: int = DEFAULT_MIN_I,
    min_J: int = DEFAULT_MIN_J,
    min_K: int = DEFAULT_MIN_K,
    disc_method: str = DEFAULT_DISC_METHOD,
    n_bins: int = DEFAULT_N_BINS,
    mv_method: str | None = DEFAULT_MV_METHOD,
    spm_algo: str = DEFAULT_SPM_ALGO,
    time_relaxed: bool = DEFAULT_TIME_RELAXED,
    coherence_threshold: float = DEFAULT_COHERENCE_THRESHOLD,
    overlap_filter: float | None = DEFAULT_OVERLAP_FILTER,
    jar_path: str | None = None,
    keep_temp_files: bool = False,
) -> dict:
    """
    Full TriHSPAM weather pipeline from raw daily history.

    Input hist_df must contain:
        - date
        - tmin
        - tmax
        - tavg
    """
    window_size = _clean_positive_int(window_size, "window_size")
    stride = _clean_positive_int(stride, "stride")
    min_I = _clean_positive_int(min_I, "min_I")
    min_J = _clean_positive_int(min_J, "min_J")
    min_K = _clean_positive_int(min_K, "min_K")
    n_bins = _clean_positive_int(n_bins, "n_bins")

    print("Creating Daily df")
    daily_df = build_enriched_daily_features(hist_df)
    print("Daily Df created")
    
    print("Creating Windows metadata and df")
    windows_meta, windows_df = build_weather_windows(
        daily_df=daily_df,
        window_size=window_size,
        stride=stride,
    )
    print("Windows metadata and df created")
    print(windows_meta)
    
    print("creating cube")
    cube_info = build_trihspam_cube(
        windows_df=windows_df,
        feature_columns=FEATURE_COLUMNS_V1,
        numeric_features=NUMERIC_FEATURES_V1,
        symbolic_features=SYMBOLIC_FEATURES_V1,
        window_size=window_size,
    )
    print("cube created")
    print(cube_info)

    print("getting config")
    config = TriHSPAMConfig(
        min_I=min_I,
        min_J=min_J,
        min_K=min_K,
        disc_method=disc_method,
        n_bins=n_bins,
        mv_method=mv_method,
        spm_algo=spm_algo,
        time_relaxed=time_relaxed,
        coherence_threshold=float(coherence_threshold),
        overlap_filter=overlap_filter,
        jar_path=jar_path,
        keep_temp_files=keep_temp_files,
    )
    print("config ready",config)

    print("getting result")
    tri_result = run_weather_trihspam(
        cube_info=cube_info,
        config=config,
    )
    print("Result : ",tri_result)

    return {
        "method": "TriHSPAM",
        "window_size": int(window_size),
        "stride": int(stride),
        "daily_rows": int(len(daily_df)),
        "n_windows": int(cube_info["n_windows"]),
        "cube_shape": list(cube_info["cube"].shape),
        "feature_columns": list(FEATURE_COLUMNS_V1),
        "numeric_features": list(NUMERIC_FEATURES_V1),
        "symbolic_features": list(SYMBOLIC_FEATURES_V1),
        "windows_meta": windows_meta,
        "engine": tri_result["engine"],
        "config": tri_result["config"],
        "abstractions": tri_result["abstractions"],
        "triclusters": tri_result["triclusters"],
        # Optional debug payloads for development
        "debug": {
            "daily_feature_columns": daily_df.columns.tolist(),
            "windows_df_columns": windows_df.columns.tolist(),
        },
    }