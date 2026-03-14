from pymongo import MongoClient
from typing import Optional
import os

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://admin:secret@localhost:27017")
DB_NAME = os.environ.get("MONGO_DB_NAME", "weather_insights")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
insights_collection = db["insights"]
