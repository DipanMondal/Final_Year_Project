import sqlite3
from pathlib import Path
import pandas as pd

import json
import datetime as _dt

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
    

def upsert_analysis_daily(city_key: str, df: pd.DataFrame):
    # df columns must match analysis_daily fields (except city)
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"]).dt.date.astype(str)
    x["city"] = city_key

    cols = [
        "city","date","tmin","tmax","tavg",
        "diurnal_range","delta_1","delta_7","roll_mean_7","roll_std_7",
        "anomaly_z","doy_sin","doy_cos","time_idx"
    ]
    x = x[cols]

    con = _connect()
    cur = con.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO analysis_daily
        (city,date,tmin,tmax,tavg,diurnal_range,delta_1,delta_7,roll_mean_7,roll_std_7,
         anomaly_z,doy_sin,doy_cos,time_idx)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [tuple(r) for r in x.itertuples(index=False, name=None)]
    )
    con.commit()
    con.close()
    return len(x)

def upsert_analysis_monthly(city_key: str, dfm: pd.DataFrame):
    x = dfm.copy()
    x["city"] = city_key
    cols = ["city","year","month","tavg_mean","tavg_std","diurnal_mean","roll_std_mean","anomaly_mean","delta_1_mean"]
    x = x[cols]

    con = _connect()
    cur = con.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO analysis_monthly
        (city,year,month,tavg_mean,tavg_std,diurnal_mean,roll_std_mean,anomaly_mean,delta_1_mean)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        [tuple(r) for r in x.itertuples(index=False, name=None)]
    )
    con.commit()
    con.close()
    return len(x)

def read_analysis_monthly(city_key: str):
    con = _connect()
    df = pd.read_sql_query(
        "SELECT * FROM analysis_monthly WHERE city=? ORDER BY year, month",
        con, params=[city_key]
    )
    con.close()
    return df

def run_log_start(run_id: str, endpoint: str, city: str, params: dict):
    con = _connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO execution_runs
        (run_id, started_at, endpoint, city, status, params_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_id, _dt.datetime.utcnow().isoformat(), endpoint, city, "running", json.dumps(params))
    )
    con.commit()
    con.close()

def run_log_end(run_id: str, status: str, duration_ms: int, result: dict | None = None, error: str | None = None):
    con = _connect()
    cur = con.cursor()
    cur.execute(
        """
        UPDATE execution_runs
        SET finished_at=?, status=?, duration_ms=?, result_json=?, error=?
        WHERE run_id=?
        """,
        (_dt.datetime.utcnow().isoformat(), status, duration_ms,
         json.dumps(result) if result is not None else None,
         error, run_id)
    )
    con.commit()
    con.close()

def list_runs(limit: int = 30):
    con = _connect()
    df = pd.read_sql_query(
        "SELECT run_id, started_at, finished_at, endpoint, city, status, duration_ms, error FROM execution_runs ORDER BY started_at DESC LIMIT ?",
        con, params=[limit]
    )
    con.close()
    return df

