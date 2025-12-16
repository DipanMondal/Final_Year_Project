from flask import Blueprint, request, jsonify
from flask_cors import CORS
import pandas as pd

from .services.openmeteo import geocode, fetch_daily
from .services.db import upsert_weather_daily, upsert_city_metadata, fetch_history, list_cities
from .services.utils import city_key
from .services.sarimax_train import train_city_sarimax
from .services.sarimax_forecast import ModelRegistry
from .services.analysis_pipeline import run_city_analysis
from .services.db import list_runs

api = Blueprint("api", __name__)
CORS(api)
registry = ModelRegistry()

@api.get("/health")
def health():
    return jsonify({"status": "ok"})

# ------------ Cities --------------

@api.get("/cities")
def cities():
    df = list_cities()
    return jsonify({"count": int(len(df)), "cities": df.to_dict(orient="records")})

@api.post("/cities")
def add_city():
    body = request.get_json(silent=True) or {}
    city = (body.get("city") or "").strip()
    country_code = (body.get("country_code") or "").strip() or None

    # keep default reasonable (faster + fewer convergence issues)
    start = (body.get("start") or "2016-01-01").strip()
    end = (body.get("end") or "2024-12-31").strip()

    if not city:
        return jsonify({"error": "city is required"}), 400

    try:
        info = geocode(city, country_code)
        key = city_key(info["name"], info.get("country_code") or country_code)

        df = fetch_daily(info["latitude"], info["longitude"], start, end)
        if df.empty:
            return jsonify({"error": "No data returned for this city/date range"}), 400

        rows = upsert_weather_daily(key, df)
        upsert_city_metadata(
            city_key=key,
            latitude=info["latitude"],
            longitude=info["longitude"],
            source="open-meteo-archive",
            start_date=start,
            end_date=end,
        )

        # Train SARIMAX using only tavg series
        y_raw = pd.Series(df["tavg"].values, index=df["date"])
        meta = train_city_sarimax(key, y_raw=y_raw, fourier_K=3)

        registry.invalidate(key)

        return jsonify({"status": "ok", "city_key": key, "rows_inserted": rows, "model_meta": meta})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------ History --------------

@api.get("/history")
def history():
    city = (request.args.get("city") or "").strip()
    start = request.args.get("start")
    end = request.args.get("end")
    country_code = (request.args.get("country_code") or "").strip() or None

    if not city:
        return jsonify({"error": "city is required"}), 400

    key = city_key(city, country_code)
    df = fetch_history(key, start, end)
    return jsonify({"city": key, "count": int(len(df)), "rows": df.to_dict(orient="records")})

# ------------ Forecast --------------

@api.get("/forecast")
def forecast():
    city = (request.args.get("city") or "").strip()
    country_code = (request.args.get("country_code") or "").strip() or None
    horizon = request.args.get("horizon", "7")

    if not city:
        return jsonify({"error": "city is required"}), 400

    try:
        horizon = int(horizon)
        horizon = max(1, min(horizon, 30))
    except:
        return jsonify({"error": "horizon must be an integer"}), 400

    key = city_key(city, country_code)

    try:
        forecaster = registry.get(key)
        return jsonify(forecaster.forecast(horizon))
    except FileNotFoundError:
        return jsonify({"error": f"City '{key}' not trained yet. Call POST /cities first."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
# ------------ Analysis --------------
        
@api.post("/analyse/<city>")
def analyse_city(city):
    country_code = (request.args.get("country_code") or "").strip() or None
    start = (request.args.get("start") or "2016-01-01").strip()
    end = (request.args.get("end") or "2024-12-31").strip()

    auto_ingest = (request.args.get("auto_ingest") or "1").strip() == "1"

    k_years = int(request.args.get("k_years") or "3")
    k_months = int(request.args.get("k_months") or "3")

    try:
        out = run_city_analysis(
            city=city,
            country_code=country_code,
            start=start,
            end=end,
            auto_ingest=auto_ingest,
            k_years=k_years,
            k_months=k_months
        )
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------ Log Runs --------------

@api.get("/runs")
def runs():
    limit = int(request.args.get("limit") or "30")
    df = list_runs(limit=max(1, min(limit, 200)))
    return jsonify({"count": int(len(df)), "runs": df.to_dict(orient="records")})

