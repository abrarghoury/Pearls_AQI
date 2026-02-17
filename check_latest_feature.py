# check_latest_feature.py
from pymongo import MongoClient
from config.settings import settings
from config.constants import (
    RAW_COLLECTION,
    CLEAN_COLLECTION,
    FEATURE_COLLECTION,
    CLEANED_FEATURE_COLLECTION,
    MODEL_COLLECTION,
    PREDICTION_COLLECTION,
    LATEST_PREDICTION_COLLECTION
)
from pprint import pprint
from datetime import datetime

# List of collections to check
COLLECTIONS_TO_CHECK = [
    RAW_COLLECTION,
    CLEAN_COLLECTION,
    FEATURE_COLLECTION,
    CLEANED_FEATURE_COLLECTION,
    MODEL_COLLECTION,
    PREDICTION_COLLECTION,
    LATEST_PREDICTION_COLLECTION
]

def check_latest_docs():
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]

    for coll_name in COLLECTIONS_TO_CHECK:
        print(f"\n=== Checking Latest Document in '{coll_name}' ===")
        collection = db[coll_name]

        # Sort by logical timestamp field for each collection
        sort_field = None
        if coll_name in [RAW_COLLECTION, CLEAN_COLLECTION, FEATURE_COLLECTION, CLEANED_FEATURE_COLLECTION]:
            sort_field = "timestamp"
        elif coll_name == MODEL_COLLECTION:
            sort_field = "trained_at"
        elif coll_name in [PREDICTION_COLLECTION, LATEST_PREDICTION_COLLECTION]:
            sort_field = "predicted_at"

        if not sort_field:
            print(f"⚠️ No sort field defined for collection '{coll_name}'. Skipping.")
            continue

        latest_doc_cursor = collection.find().sort(sort_field, -1).limit(1)
        latest_docs = list(latest_doc_cursor)

        if not latest_docs:
            print("No documents found.")
            continue

        doc = latest_docs[0]
        # Pick relevant keys dynamically
        keys_to_show = ["_id", sort_field, "timestamp", "aqi", "city", "model_name"]
        pprint({k: doc.get(k) for k in keys_to_show if k in doc})

        # Check if timestamp is today (for feature/raw/prediction)
        date_field = None
        if coll_name in [RAW_COLLECTION, CLEAN_COLLECTION, FEATURE_COLLECTION, CLEANED_FEATURE_COLLECTION]:
            date_field = "timestamp"
        elif coll_name in [PREDICTION_COLLECTION, LATEST_PREDICTION_COLLECTION]:
            date_field = "predicted_at"

        if date_field and doc.get(date_field):
            doc_date = doc.get(date_field)
            if isinstance(doc_date, datetime):
                if doc_date.date() < datetime.utcnow().date():
                    print("⚠️ Warning: Latest document is older than today. Pipeline may not have run properly.")
                else:
                    print("✅ Latest document is up-to-date with today.")

if __name__ == "__main__":
    check_latest_docs()
