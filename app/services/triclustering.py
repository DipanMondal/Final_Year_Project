import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

FEATURES = ["tavg_mean","tavg_std","diurnal_mean","roll_std_mean","anomaly_mean","delta_1_mean"]

def tricluster_year_month_features(monthly: pd.DataFrame, k_years: int = 3, k_months: int = 3):
    # monthly columns: year, month, FEATURES...
    if monthly.empty:
        raise ValueError("No monthly analysis data available for triclustering.")

    years = sorted(monthly["year"].unique().tolist())
    months = list(range(1, 13))

    # Build tensor Y x 12 x F
    Y = len(years)
    F = len(FEATURES)
    T = np.full((Y, 12, F), np.nan, dtype=float)

    year_to_i = {y:i for i,y in enumerate(years)}
    for _, r in monthly.iterrows():
        yi = year_to_i[int(r["year"])]
        mi = int(r["month"]) - 1
        T[yi, mi, :] = [float(r[f]) for f in FEATURES]

    # fill missing with feature mean
    flat = T.reshape(-1, F)
    col_means = np.nanmean(flat, axis=0)
    inds = np.where(np.isnan(flat))
    flat[inds] = np.take(col_means, inds[1])
    T = flat.reshape(Y, 12, F)

    # z-score normalize per feature
    flat2 = T.reshape(-1, F)
    mu = flat2.mean(axis=0)
    sd = flat2.std(axis=0)
    sd[sd == 0] = 1.0
    flat2 = (flat2 - mu) / sd
    Z = flat2.reshape(Y, 12, F)

    ky = min(k_years, Y) if Y >= 2 else 1
    X_year = Z.reshape(Y, 12 * F)

    year_labels = KMeans(n_clusters=ky, n_init=10, random_state=42).fit_predict(X_year) if ky > 1 else np.zeros(Y, dtype=int)

    clusters = []
    for yc in range(ky):
        y_idx = np.where(year_labels == yc)[0]
        years_in = [years[i] for i in y_idx.tolist()]

        # mean pattern for this year cluster: 12 x F
        M = Z[y_idx].mean(axis=0)  # (12,F)

        km = min(k_months, 12) if 12 >= 2 else 1
        month_labels = KMeans(n_clusters=km, n_init=10, random_state=42).fit_predict(M) if km > 1 else np.zeros(12, dtype=int)

        for mc in range(km):
            m_idx = np.where(month_labels == mc)[0]
            months_in = [int(i+1) for i in m_idx.tolist()]

            sig = M[m_idx].mean(axis=0)  # (F,)
            order = np.argsort(np.abs(sig))[::-1][:5]
            signature = []
            for j in order:
                signature.append({
                    "feature": FEATURES[j],
                    "zscore": float(sig[j]),
                    "direction": "high" if sig[j] >= 0 else "low"
                })

            clusters.append({
                "years": years_in,
                "months": months_in,
                "signature_top5": signature
            })

    return {
        "features_used": FEATURES,
        "k_years": int(ky),
        "k_months": int(k_months),
        "clusters": clusters
    }
