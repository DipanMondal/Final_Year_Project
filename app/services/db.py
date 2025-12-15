import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("data/weather.db")

def fetch_history(city: str, start: str | None, end: str | None):
    con = sqlite3.connect(DB_PATH)
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
