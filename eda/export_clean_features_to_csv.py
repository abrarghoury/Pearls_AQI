# =========================================================
# EXPORT CLEANED FEATURES TO CSV
# ---------------------------------------------------------
# Loads: CLEANED_FEATURE_COLLECTION (MongoDB Atlas)
# Saves: cleaned_features.csv (local laptop)
# ---------------------------------------------------------
# Safe:
# - Does NOT modify MongoDB
# - Only reads
# =========================================================

import os
import pandas as pd

from config.mongo import get_database
from config.constants import CLEANED_FEATURE_COLLECTION
from config.logging import logger


def export_clean_features_to_csv(
        filename: str = "cleaned_features.csv",
        folder: str = "eda_data"
):
    logger.info("========== EXPORT CLEAN FEATURES STARTED ==========")

    db = get_database()
    col = db[CLEANED_FEATURE_COLLECTION]

    data = list(col.find({}, {"_id": 0}))

    if not data:
        raise ValueError("CLEANED_FEATURE_COLLECTION is empty!")

    df = pd.DataFrame(data)

    logger.info(f"Rows loaded from MongoDB: {len(df)}")
    logger.info(f"Columns: {len(df.columns)}")

    # -----------------------------------------------------
    # Create folder if not exists
    # -----------------------------------------------------
    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, filename)

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------
    df.to_csv(filepath, index=False)

    logger.info(f"CSV saved at: {filepath}")
    logger.info("========== EXPORT COMPLETE ==========")

    print("\n✅ CLEAN FEATURES EXPORTED SUCCESSFULLY")
    print(f"Location: {filepath}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    export_clean_features_to_csv()
