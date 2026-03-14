import time
import uuid
import pandas as pd
import logging

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
from .analysis_features import build_daily_analysis_features, build_monthly_analysis
from .triclustering import tricluster_year_month_features

from .insights import compute_insights_payload
from .mongo_client import insights_collection


logger = logging.getLogger(__name__)

def run_city_analysis(
    city: str,
    country_code: str | None,
    start: str,
    end: str,
    auto_ingest: bool = True,
    k_years: int = 3,
    k_months: int = 3
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
    }

    # store run in SQLite DB (metadata only)
    run_log_start(run_id, endpoint="/analyse/<city>", city=key, params=params)
    upsert_insights_cache(
        city_key=key,
        analysis_run_id=run_id,
        data_start=start,
        data_end=end,
        status="running",
        mongo_id=None,
        error=None,
        version=1
    )

    # bind run_id to log lines
    with bind_run_id(run_id):
        logger.info(
            f"ANALYSE_START city={city} key={key} country_code={country_code} "
            f"start={start} end={end} auto_ingest={auto_ingest} k_years={k_years} k_months={k_months}"
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
                _step_end("auto_ingest", {"rows_after": int(len(hist)), "inserted": int(inserted)})

            # 3) Stop if still empty
            if hist.empty:
                raise ValueError(f"No history for '{key}'. Ingest via POST /cities or use auto_ingest=1.")

            # 4) Build daily analysis features
            _step_start("build_daily_features")
            daily_feat = build_daily_analysis_features(hist)
            _step_end("build_daily_features", {"daily_feat_rows": int(len(daily_feat))})

            # 5) Store daily features
            _step_start("store_analysis_daily")
            daily_rows = upsert_analysis_daily(key, daily_feat)
            _step_end("store_analysis_daily", {"upserted": int(daily_rows)})

            # 6) Build monthly analysis
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
            tri = tricluster_year_month_features(monthly_db, k_years=k_years, k_months=k_months)
            _step_end("triclustering", {"clusters": len(tri.get("clusters", [])) if isinstance(tri, dict) else 0})

            # 10) Compute insights
            _step_start("compute_insights")
            insights_payload = compute_insights_payload(
                city_key=key,
                daily_feat_df=daily_feat,
                monthly_df=monthly_db,
                tri=tri,
                run_id=run_id
            )
            _step_end("compute_insights", {"insights_keys": list(insights_payload.keys())})

            # 11) Store insights in MongoDB
            _step_start("store_insights_mongo")
            mongo_res = insights_collection.replace_one(
                {"city_key": key},
                insights_payload,
                upsert=True
            )
            mongo_id = str(mongo_res.upserted_id) if mongo_res.upserted_id else str(
                insights_collection.find_one({"city_key": key})["_id"]
            )
            _step_end("store_insights_mongo", {"mongo_id": mongo_id})

            # 12) Update SQLite cache (metadata only)
            upsert_insights_cache(
                city_key=key,
                analysis_run_id=run_id,
                data_start=insights_payload.get("data_start"),
                data_end=insights_payload.get("data_end"),
                status="ok",
                mongo_id=mongo_id,
                error=None,
                version=1
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
            run_log_end(run_id, status="ok", duration_ms=total_ms, result=result, error=None)
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
                version=1
            )
            total_ms = int((time.time() - pipeline_t0) * 1000)
            logger.exception(f"ANALYSE_END status=error total_ms={total_ms} err={e}")
            run_log_end(run_id, status="error", duration_ms=total_ms, result=None, error=str(e))
            raise

