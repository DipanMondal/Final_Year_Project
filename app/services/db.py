import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("data/weather.db")

def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def upsert_weather_daily(city_key: str, df: pd.DataFrame):
    df2 = df.copy()
    df2["date"] = df2["date"].dt.date.astype(str)

    con = _connect()
    cur = con.cursor()
    rows = [(city_key, r["date"], float(r["tmin"]), float(r["tmax"]), float(r["tavg"])) for _, r in df2.iterrows()]
    cur.executemany(
        "INSERT OR REPLACE INTO weather_daily (city,date,tmin,tmax,tavg) VALUES (?,?,?,?,?)",
        rows
    )
    con.commit()
    con.close()
    return len(rows)

def upsert_city_metadata(city_key: str, latitude: float, longitude: float, source: str, start_date: str, end_date: str):
    con = _connect()
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO metadata (city, latitude, longitude, source, start_date, end_date)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (city_key, latitude, longitude, source, start_date, end_date))
    con.commit()
    con.close()

def fetch_history(city: str, start: str | None, end: str | None):
    con = _connect()
    q = "SELECT date,tmin,tmax,tavg FROM weather_daily WHERE city=?"
    params = [city]
    if start:
        q += " AND date >= ?"
        params.append(start)
    if end:
        q += " AND date <= ?"
        params.append(end)
    q += " ORDER BY date ASC"
    df = pd.read_sql_query(q, con, params=params)
    con.close()
    return df

def list_cities():
    con = _connect()
    df = pd.read_sql_query(
        "SELECT city, latitude, longitude, source, start_date, end_date FROM metadata ORDER BY city",
        con
    )
    con.close()
    return df
