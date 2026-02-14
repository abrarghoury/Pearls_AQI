from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from config.settings import settings
from config.logging import logger


class MongoManager:
    """
    Central MongoDB connection manager.
    Ensures a single reusable client across the project.
    """

    _client = None
    _db = None

    @classmethod
    def connect(cls):
        if cls._client is None:
            try:
                logger.info("Connecting to MongoDB Atlas...")

                cls._client = MongoClient(
                    settings.MONGO_URI,
                    serverSelectionTimeoutMS=5000
                )

                # Force connection test
                cls._client.admin.command("ping")

                cls._db = cls._client[settings.MONGO_DB_NAME]

                logger.info(f"Connected to MongoDB | DB: {settings.MONGO_DB_NAME}")

            except ConnectionFailure as e:
                logger.exception("MongoDB connection failed.")
                raise e

        return cls._db


def get_database():
    """
    Helper function for pipelines.
    Usage:
        db = get_database()
    """
    return MongoManager.connect()
