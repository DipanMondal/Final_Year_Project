import sqlite3
from pathlib import Path

DB_PATH = Path("data/weather.db")

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS weather_daily (
        city TEXT NOT NULL,
        date TEXT NOT NULL,
        tmin REAL,
        tmax REAL,
        tavg REAL,
        PRIMARY KEY (city, date)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        city TEXT PRIMARY KEY,
        latitude REAL,
        longitude REAL,
        source TEXT,
        start_date TEXT,
        end_date TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS analysis_daily (
        city TEXT NOT NULL,
        date TEXT NOT NULL,
        tmin REAL,
        tmax REAL,
        tavg REAL,
        diurnal_range REAL,
        delta_1 REAL,
        delta_7 REAL,
        roll_mean_7 REAL,
        roll_std_7 REAL,
        anomaly_z REAL,
        doy_sin REAL,
        doy_cos REAL,
        time_idx REAL,
        PRIMARY KEY (city, date)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS analysis_monthly (
        city TEXT NOT NULL,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL,
        tavg_mean REAL,
        tavg_std REAL,
        diurnal_mean REAL,
        roll_std_mean REAL,
        anomaly_mean REAL,
        delta_1_mean REAL,
        PRIMARY KEY (city, year, month)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS execution_runs (
        run_id TEXT PRIMARY KEY,
        started_at TEXT,
        finished_at TEXT,
        endpoint TEXT,
        city TEXT,
        status TEXT,
        duration_ms INTEGER,
        params_json TEXT,
        result_json TEXT,
        error TEXT
    );
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS insights_cache (
        city TEXT PRIMARY KEY,
        updated_at TEXT,
        analysis_run_id TEXT,
        data_start TEXT,
        data_end TEXT,
        status TEXT,                 -- ok | running | error
        payload_json TEXT,
        error TEXT,
        version INTEGER
    );
    """)

    con.commit()
    con.close()
    print(f"Initialized DB schema at {DB_PATH}")

if __name__ == "__main__":
    main()
