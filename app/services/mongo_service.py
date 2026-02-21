from pymongo import MongoClient
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

    # -------------------------------------------------
    # SINGLETON DB CONNECTION
    # -------------------------------------------------
    @classmethod
    def get_db(cls):
        if cls._client is None:
            cls._client = MongoClient(
                settings.MONGO_URI,
                serverSelectionTimeoutMS=5000,   # Prevent long hanging
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                retryWrites=True
            )
            cls._db = cls._client[settings.MONGO_DB_NAME]
        return cls._db

    # -------------------------------------------------
    # LATEST FEATURE ROW (CLEANED FEATURE)
    # -------------------------------------------------
    @classmethod
    def get_latest_features(cls):
        """
        Returns the most recent cleaned feature row.
        Ensures dashboard shows latest AQI as models were trained.
        """
        db = cls.get_db()
        return db[CLEANED_FEATURE_COLLECTION].find_one(
            sort=[("timestamp", -1)],
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
