from pymongo import MongoClient
from config.settings import settings
from config.constants import (
    RAW_COLLECTION,
    CLEAN_COLLECTION,
    FEATURE_COLLECTION,
    PREDICTION_COLLECTION,
    MODEL_COLLECTION
)


class MongoService:
    _client = None
    _db = None

    # -------------------------------------------------
    # SINGLETON DB CONNECTION
    # -------------------------------------------------
    @classmethod
    def get_db(cls):
        if cls._client is None:
            cls._client = MongoClient(settings.MONGO_URI)
            cls._db = cls._client[settings.MONGO_DB_NAME]
        return cls._db

    # -------------------------------------------------
    # LATEST FEATURE ROW
    # -------------------------------------------------
    @classmethod
    def get_latest_features(cls):
        """
        Returns the most recent feature row based on timestamp.
        This ensures dashboard shows latest AQI correctly.
        """
        db = cls.get_db()
        return db[FEATURE_COLLECTION].find_one(
            sort=[("timestamp", -1)],  # 🔑 Corrected: use actual timestamp
            projection={"_id": 0}
        )

    # -------------------------------------------------
    # LAST PREDICTION LOG
    # -------------------------------------------------
    @classmethod
    def get_latest_prediction_log(cls):
        db = cls.get_db()
        return db[PREDICTION_COLLECTION].find_one(
            sort=[("created_at", -1)],
            projection={"_id": 0}
        )

    # -------------------------------------------------
    # MODEL REGISTRY
    # -------------------------------------------------
    @classmethod
    def get_model_registry(cls):
        db = cls.get_db()
        return list(
            db[MODEL_COLLECTION].find({}, {"_id": 0})
        )

    # -------------------------------------------------
    # LATEST RAW WEATHER
    # -------------------------------------------------
    @classmethod
    def get_latest_raw(cls):
        db = cls.get_db()
        return db[RAW_COLLECTION].find_one(
            sort=[("timestamp", -1)],
            projection={"_id": 0}
        )

    # =================================================
    # ⭐ NEW — AQI TREND (VERY IMPORTANT FOR DASHBOARD)
    # =================================================
    @classmethod
    def get_recent_features(cls, limit=48):
        """
        Returns last N AQI feature rows for trend charts.
        Default = 48 hours.
        """
        db = cls.get_db()

        data = list(
            db[FEATURE_COLLECTION]
            .find(
                {},
                {
                    "_id": 0,
                    "timestamp": 1,
                    "aqi": 1
                }
            )
            .sort("timestamp", -1)
            .limit(limit)
        )

        # Reverse so chart goes oldest → newest
        return list(reversed(data))
