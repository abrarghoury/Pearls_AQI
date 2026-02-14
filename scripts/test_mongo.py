from config.mongo import get_database
from config.logging import logger


def test_connection():
    try:
        db = get_database()

        logger.info("MongoDB connection test successful.")
        print(f"Connected to database: {db.name}")

    except Exception as e:
        logger.exception("MongoDB test failed.")
        print("Connection failed:", str(e))


if __name__ == "__main__":
    test_connection()
