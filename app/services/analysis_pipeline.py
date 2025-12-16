import time
import uuid
import pandas as pd

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
    run_log_end
)
from .analysis_features import build_daily_analysis_features, build_monthly_analysis
from .triclustering import tricluster_year_month_features

def run_city_analysis(city: str, country_code: str | None, start: str, end: str, auto_ingest: bool = True,
                     k_years: int = 3, k_months: int = 3):
    run_id = uuid.uuid4().hex
    t0 = time.time()

    key = city_key(city, country_code)
    params = {
        "city": city, "country_code": country_code, "start": start, "end": end,
        "auto_ingest": auto_ingest, "k_years": k_years, "k_months": k_months
    }
    run_log_start(run_id, endpoint="/analyse/<city>", city=key, params=params)

    try:
        hist = fetch_history(key, None, None)

        # auto-ingest if missing
        if hist.empty and auto_ingest:
            info = geocode(city, country_code)
            key = city_key(info["name"], info.get("country_code") or country_code)

            df = fetch_daily(info["latitude"], info["longitude"], start, end)
            if df.empty:
                raise ValueError("No data returned from Open-Meteo for this city/date range.")

            upsert_weather_daily(key, df)
            upsert_city_metadata(
                city_key=key,
                latitude=info["latitude"],
                longitude=info["longitude"],
                source="open-meteo-archive",
                start_date=start,
                end_date=end,
            )
            hist = fetch_history(key, None, None)

        if hist.empty:
            raise ValueError(f"No history for '{key}'. Either ingest via POST /cities or call analyse with auto_ingest=1.")

        daily_feat = build_daily_analysis_features(hist)
        daily_rows = upsert_analysis_daily(key, daily_feat)

        monthly_feat = build_monthly_analysis(daily_feat)
        monthly_rows = upsert_analysis_monthly(key, monthly_feat)

        monthly_db = read_analysis_monthly(key)
        tri = tricluster_year_month_features(monthly_db, k_years=k_years, k_months=k_months)

        result = {
            "run_id": run_id,
            "city_key": key,
            "analysis_daily_rows": int(daily_rows),
            "analysis_monthly_rows": int(monthly_rows),
            "triclustering": tri
        }

        dt_ms = int((time.time() - t0) * 1000)
        run_log_end(run_id, status="ok", duration_ms=dt_ms, result=result, error=None)
        return result

    except Exception as e:
        dt_ms = int((time.time() - t0) * 1000)
        run_log_end(run_id, status="error", duration_ms=dt_ms, result=None, error=str(e))
        raise
