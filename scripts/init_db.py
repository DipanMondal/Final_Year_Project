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

    con.commit()
    con.close()
    print(f"Initialized DB at {DB_PATH}")

if __name__ == "__main__":
    main()
