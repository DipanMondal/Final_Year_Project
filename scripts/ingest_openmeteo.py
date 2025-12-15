import argparse
import pandas as pd
import requests
from pathlib import Path

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

def geocode(city: str, country_code: str | None = None):
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    if country_code:
        params["country_code"] = country_code
    r = requests.get(GEOCODE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "results" not in data or not data["results"]:
        raise ValueError(f"City not found: {city}")
    x = data["results"][0]
    return float(x["latitude"]), float(x["longitude"]), x.get("name", city), x.get("country", "")

def fetch_daily(lat: float, lon: float, start: str, end: str):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"],
        "timezone": "auto"
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
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--country_code", default=None, help="Optional, like IN")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default="data/raw_weather.csv")
    args = ap.parse_args()

    lat, lon, name, country = geocode(args.city, args.country_code)
    df = fetch_daily(lat, lon, args.start, args.end)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        f'{{"city":"{name}","country":"{country}","latitude":{lat},"longitude":{lon},"start":"{args.start}","end":"{args.end}"}}',
        encoding="utf-8"
    )

    print(f"Saved: {out_path}  rows={len(df)}")
    print(f"Meta : {meta_path}")

if __name__ == "__main__":
    main()
