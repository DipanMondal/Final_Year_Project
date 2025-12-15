from flask import Blueprint, request, jsonify
from flask_cors import CORS
from .services.db import fetch_history
from .services.model_service import TempForecaster

api = Blueprint("api", __name__)
CORS(api)

# this will be set from create_app()
forecaster: TempForecaster | None = None

def init_services():
    global forecaster
    if forecaster is None:
        forecaster = TempForecaster()

@api.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": forecaster is not None})

@api.get("/history")
def history():
    city = request.args.get("city", "").strip()
    start = request.args.get("start")
    end = request.args.get("end")
    if not city:
        return jsonify({"error": "city is required"}), 400

    df = fetch_history(city, start, end)
    return jsonify({
        "city": city,
        "count": int(len(df)),
        "rows": df.to_dict(orient="records")
    })

@api.get("/forecast")
def forecast():
    city = request.args.get("city", "").strip()
    horizon = request.args.get("horizon", "7")
    if not city:
        return jsonify({"error": "city is required"}), 400

    try:
        horizon = int(horizon)
        horizon = max(1, min(horizon, 30))
    except:
        return jsonify({"error": "horizon must be an integer"}), 400

    try:
        out = forecaster.forecast(city, horizon)  # type: ignore
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
