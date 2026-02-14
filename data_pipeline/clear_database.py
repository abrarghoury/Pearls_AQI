from pymongo import MongoClient
from config.settings import settings


def clear_database():
    """
    Drops the entire Mongo database safely.
    Use ONLY when resetting pipeline.
    """

    client = MongoClient(settings.MONGO_URI)

    db_name = settings.MONGO_DB_NAME

    if db_name in client.list_database_names():
        client.drop_database(db_name)
        print(f"✅ Database '{db_name}' deleted successfully.")
    else:
        print(f"⚠️ Database '{db_name}' does not exist.")


if __name__ == "__main__":
    confirm = input(
        "Are you sure you want to DELETE the entire database? Type 'YES' to continue: "
    )

    if confirm == "YES":
        clear_database()
    else:
        print("❌ Operation cancelled.")
