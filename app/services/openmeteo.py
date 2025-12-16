import requests
import pandas as pd
import logging
from .logging_utils import trace

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

def geocode(city: str, country_code: str | None = None):
    logger.debug(f"GEOCODE request city={city} country_code={country_code}")
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    if country_code:
        params["country_code"] = country_code

    r = requests.get(GEOCODE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        raise ValueError(f"City not found: {city}")

    x = data["results"][0]
    logger.debug(f"GEOCODE result lat={...} lon={...}")
    return {
        "name": x.get("name", city),
        "country": x.get("country", ""),
        "country_code": x.get("country_code", country_code or ""),
        "latitude": float(x["latitude"]),
        "longitude": float(x["longitude"]),
    }

@trace
def fetch_daily(lat: float, lon: float, start: str, end: str):
    logger.debug(f"ARCHIVE request lat={lat} lon={lon} start={start} end={end}")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"],
        "timezone": "auto",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    d = j.get("daily", {})

    df = pd.DataFrame({
        "date": d.get("time", []),
        "tmax": d.get("temperature_2m_max", []),
        "tmin": d.get("temperature_2m_min", []),
        "tavg": d.get("temperature_2m_mean", []),
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna()
    logger.info(f"ARCHIVE rows_returned={len(df)}")
    return df
