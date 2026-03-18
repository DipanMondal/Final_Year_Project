import time
import uuid
import logging
import pandas as pd

from ..log_context import bind_run_id
from .utils import city_key
from .openmeteo import geocode, fetch_daily
from .db import (
    fetch_history,
    upsert_weather_daily,
    upsert_city_metadata,
    upsert_analysis_daily,
    upsert_analysis_monthly,
    read_analysis_monthly,
    run_log_start,
    run_log_end,
    upsert_insights_cache,
)
from .analysis_features import (
    build_daily_analysis_features,
    build_monthly_analysis,
)
from .triclustering import (
    tricluster_year_month_features,
    run_weather_triclustering_from_history,
)
from .insights import compute_insights_payload

import os
import json
from pathlib import Path
from datetime import datetime


logger = logging.getLogger(__name__)


INSIGHTS_JSON_DIR = Path("artifacts/insights_json")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def write_insights_payload_to_json(
    city_key: str,
    run_id: str,
    payload: dict,
    base_dir: Path = INSIGHTS_JSON_DIR,
) -> str:
    """
    Save full insights payload to local JSON file and return the file path as string.
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_city = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in city_key)
    filename = f"{safe_city}__{run_id}__{ts}.json"
    file_path = base_dir / filename

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)

    return str(file_path)


def _safe_len(x) -> int:
    try:
        return int(len(x))
    except Exception:
        return 0


def run_city_analysis(
    city: str,
    country_code: str | None,
    start: str,
    end: str,
    auto_ingest: bool = True,
    k_years: int = 3,
    k_months: int = 3,
    use_legacy_triclustering: bool = False,
    window_size: int = 14,
    stride: int = 14,
    min_I: int = 8,
    min_J: int = 2,
    min_K: int = 2,
    disc_method: str = "eq_size",
    n_bins: int = 3,
    mv_method: str | None = None,
    spm_algo: str = "fournier08closed",
    time_relaxed: bool = False,
    coherence_threshold: float = 0.4,
    overlap_filter: float | None = 0.7,
    jar_path: str | None = None,
    keep_temp_files: bool = False,
):
    run_id = uuid.uuid4().hex
    pipeline_t0 = time.time()

    key = city_key(city, country_code)
    params = {
        "city": city,
        "country_code": country_code,
        "start": start,
        "end": end,
        "auto_ingest": auto_ingest,
        "k_years": k_years,
        "k_months": k_months,
        "use_legacy_triclustering": use_legacy_triclustering,
        "window_size": window_size,
        "stride": stride,
        "min_I": min_I,
        "min_J": min_J,
        "min_K": min_K,
        "disc_method": disc_method,
        "n_bins": n_bins,
        "mv_method": mv_method,
        "spm_algo": spm_algo,
        "time_relaxed": time_relaxed,
        "coherence_threshold": coherence_threshold,
        "overlap_filter": overlap_filter,
        "jar_path": jar_path,
        "keep_temp_files": keep_temp_files,
    }

    run_log_start(run_id, endpoint="/analyse/<city>", city=key, params=params)
    upsert_insights_cache(
        city_key=key,
        analysis_run_id=run_id,
        data_start=start,
        data_end=end,
        status="running",
        mongo_id=None,
        error=None,
        version=2,
    )

    with bind_run_id(run_id):
        logger.info(
            "ANALYSE_START "
            f"city={city} key={key} country_code={country_code} "
            f"start={start} end={end} auto_ingest={auto_ingest} "
            f"use_legacy_triclustering={use_legacy_triclustering} "
            f"window_size={window_size} stride={stride} "
            f"min_I={min_I} min_J={min_J} min_K={min_K} "
            f"disc_method={disc_method} n_bins={n_bins} "
            f"mv_method={mv_method} spm_algo={spm_algo} "
            f"time_relaxed={time_relaxed} coherence_threshold={coherence_threshold} "
            f"overlap_filter={overlap_filter}"
        )

        steps = {}

        def _step_start(name: str):
            logger.debug(f"STEP_START {name}")
            steps[name] = {"t0": time.time()}

        def _step_end(name: str, extra: dict | None = None):
            dt_ms = int((time.time() - steps[name]["t0"]) * 1000)
            steps[name]["dt_ms"] = dt_ms
            if extra:
                steps[name].update(extra)
            logger.info(f"STEP_END {name} dt_ms={dt_ms} extra={extra or {}}")

        try:
            # 1) Fetch history
            _step_start("fetch_history_initial")
            hist = fetch_history(key, None, None)
            _step_end("fetch_history_initial", {"rows": int(len(hist))})

            # 2) Auto-ingest if missing
            if hist.empty and auto_ingest:
                _step_start("auto_ingest")
                info = geocode(city, country_code)
                key = city_key(info["name"], info.get("country_code") or country_code)

                df = fetch_daily(info["latitude"], info["longitude"], start, end)
                if df.empty:
                    raise ValueError("No data returned from Open-Meteo for this city/date range.")

                inserted = upsert_weather_daily(key, df)
                upsert_city_metadata(
                    city_key=key,
                    latitude=info["latitude"],
                    longitude=info["longitude"],
                    source="open-meteo-archive",
                    start_date=start,
                    end_date=end,
                )

                hist = fetch_history(key, None, None)
                _step_end(
                    "auto_ingest",
                    {
                        "rows_after": int(len(hist)),
                        "inserted": int(inserted),
                    },
                )

            # 3) Stop if still empty
            if hist.empty:
                raise ValueError(
                    f"No history for '{key}'. Ingest via POST /cities or use auto_ingest=1."
                )

            # 4) Build legacy daily analysis features for existing tables/charts
            _step_start("build_daily_features")
            daily_feat = build_daily_analysis_features(hist)
            _step_end("build_daily_features", {"daily_feat_rows": int(len(daily_feat))})

            # 5) Store daily features
            _step_start("store_analysis_daily")
            daily_rows = upsert_analysis_daily(key, daily_feat)
            _step_end("store_analysis_daily", {"upserted": int(daily_rows)})

            # 6) Build monthly analysis for existing baseline charts
            _step_start("build_monthly_features")
            monthly_feat = build_monthly_analysis(daily_feat)
            _step_end("build_monthly_features", {"monthly_feat_rows": int(len(monthly_feat))})

            # 7) Store monthly analysis
            _step_start("store_analysis_monthly")
            monthly_rows = upsert_analysis_monthly(key, monthly_feat)
            _step_end("store_analysis_monthly", {"upserted": int(monthly_rows)})

            # 8) Read monthly back
            _step_start("read_analysis_monthly")
            monthly_db = read_analysis_monthly(key)
            _step_end("read_analysis_monthly", {"rows": int(len(monthly_db))})

            # 9) Triclustering
            _step_start("triclustering")
            if use_legacy_triclustering:
                tri = tricluster_year_month_features(
                    monthly_db,
                    k_years=k_years,
                    k_months=k_months,
                )
                tri_extra = {
                    "method": "legacy_kmeans_placeholder",
                    "clusters": len(tri.get("clusters", [])) if isinstance(tri, dict) else 0,
                }
            else:
                tri = run_weather_triclustering_from_history(
                    hist_df=hist,
                    window_size=window_size,
                    stride=stride,
                    min_I=min_I,
                    min_J=min_J,
                    min_K=min_K,
                    disc_method=disc_method,
                    n_bins=n_bins,
                    mv_method=mv_method,
                    spm_algo=spm_algo,
                    time_relaxed=time_relaxed,
                    coherence_threshold=coherence_threshold,
                    overlap_filter=overlap_filter,
                    jar_path=jar_path,
                    keep_temp_files=keep_temp_files,
                )
                tri_extra = {
                    "method": tri.get("method", "TriHSPAM"),
                    "triclusters": len(tri.get("triclusters", [])),
                    "n_windows": tri.get("n_windows"),
                    "cube_shape": tri.get("cube_shape"),
                }
            _step_end("triclustering", tri_extra)

            # 10) Compute insights
            # For now we keep the same insights function signature.
            # It will continue to use monthly summaries, and later we will upgrade
            # insights.py to properly render TriHSPAM triclusters.
            _step_start("compute_insights")
            insights_payload = compute_insights_payload(
                city_key=key,
                daily_feat_df=daily_feat,
                monthly_df=monthly_db,
                tri=tri,
                run_id=run_id,
            )
            _step_end(
                "compute_insights",
                {
                    "insights_keys": list(insights_payload.keys()),
                },
            )

            # 11) Store insights as local JSON file instead of MongoDB
            _step_start("store_insights_json")
            mongo_id = write_insights_payload_to_json(
                city_key=key,
                run_id=run_id,
                payload=insights_payload,
            )
            _step_end("store_insights_json", {"mongo_id": mongo_id})

            # 12) Update SQLite cache
            upsert_insights_cache(
                city_key=key,
                analysis_run_id=run_id,
                data_start=insights_payload.get("data_start"),
                data_end=insights_payload.get("data_end"),
                status="ok",
                mongo_id=mongo_id,
                error=None,
                version=2,
            )

            # 13) Prepare result
            result = {
                "run_id": run_id,
                "city_key": key,
                "analysis_daily_rows": int(daily_rows),
                "analysis_monthly_rows": int(monthly_rows),
                "triclustering": tri,
                "step_timings_ms": {k: v.get("dt_ms") for k, v in steps.items()},
            }

            total_ms = int((time.time() - pipeline_t0) * 1000)
            logger.info(f"ANALYSE_END status=ok total_ms={total_ms}")
            run_log_end(
                run_id,
                status="ok",
                duration_ms=total_ms,
                result=result,
                error=None,
            )
            return result

        except Exception as e:
            upsert_insights_cache(
                city_key=key,
                analysis_run_id=run_id,
                data_start=start,
                data_end=end,
                status="error",
                mongo_id=None,
                error=str(e),
                version=2,
            )
            total_ms = int((time.time() - pipeline_t0) * 1000)
            logger.exception(f"ANALYSE_END status=error total_ms={total_ms} err={e}")
            run_log_end(
                run_id,
                status="error",
                duration_ms=total_ms,
                result=None,
                error=str(e),
            )
            raise