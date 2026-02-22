from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd

from config.settings import settings
from config.constants import (
    RAW_COLLECTION,
    CLEAN_COLLECTION,
    CLEANED_FEATURE_COLLECTION,
    PREDICTION_COLLECTION,
    MODEL_COLLECTION
)


class MongoService:
    _client = None
    _db = None

    # =================================================
    # SINGLETON DB CONNECTION
    # =================================================
    @classmethod
    def get_db(cls):
        if cls._client is None:
            cls._client = MongoClient(
                settings.MONGO_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                retryWrites=True
            )
            cls._db = cls._client[settings.MONGO_DB_NAME]
        return cls._db

    # =================================================
    # LATEST CLEANED FEATURE (FOR DASHBOARD KPIs)
    # =================================================
    @classmethod
    def get_latest_features(cls):
        db = cls.get_db()

        row = db[CLEANED_FEATURE_COLLECTION].find_one(
            sort=[("validation_done_at", -1)],
            projection={"_id": 0}
        )

        if not row:
            return None

        for k in ["feature_generated_at", "validation_done_at"]:
            if k in row:
                row[k] = pd.to_datetime(row[k], errors="coerce")

        return row

    # =================================================
    # LATEST PREDICTION LOG
    # =================================================
    @classmethod
    def get_latest_prediction_log(cls):
        db = cls.get_db()

        row = db[PREDICTION_COLLECTION].find_one(
            sort=[("predicted_at", -1)],
            projection={"_id": 0}
        )

        if not row:
            return None

        if "predicted_at" in row:
            row["predicted_at"] = pd.to_datetime(
                row["predicted_at"], errors="coerce"
            )

        return row

    # =================================================
    # MODEL REGISTRY
    # =================================================
    @classmethod
    def get_model_registry(cls):
        db = cls.get_db()

        data = list(
            db[MODEL_COLLECTION].find({}, {"_id": 0})
        )

        for row in data:
            for k in ["training_date", "created_at"]:
                if k in row:
                    row[k] = pd.to_datetime(row[k], errors="coerce")

        return data

    # =================================================
    # LATEST RAW AQI / WEATHER
    # =================================================
    @classmethod
    def get_latest_raw(cls):
        db = cls.get_db()

        row = db[RAW_COLLECTION].find_one(
            sort=[("timestamp", -1)],
            projection={"_id": 0}
        )

        if not row:
            return None

        if "timestamp" in row:
            row["timestamp"] = pd.to_datetime(
                row["timestamp"], errors="coerce"
            )

        return row

    # =================================================
    # ⭐ AQI TREND (CLEANED FEATURE)
    # =================================================
    @classmethod
    def get_recent_features(cls, limit: int = 48):
        """
        Returns last N cleaned AQI feature rows for trend charts.
        Default = 48 hours.
        """
        db = cls.get_db()
        data = list(
            db[CLEANED_FEATURE_COLLECTION]
            .find(
                {},
                {"_id": 0, "timestamp": 1, "aqi": 1}
            )
            .sort("timestamp", -1)
            .limit(limit)
        )

        # Reverse so chart goes oldest → newest
        return list(reversed(data))
    