import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# ==============================
# LOAD ENV
# ==============================

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise Exception("MONGO_URI not found in .env")

# ==============================
# CONNECT DB
# ==============================

client = MongoClient(MONGO_URI)

db_name = "pearls_aqi"   
db = client[db_name]

FEATURE_COLLECTION = "aqi_features"
CLEAN_COLLECTION = "feature_cleaned"

print("\n========== SCHEMA CHECK STARTED ==========\n")

# ==============================
# LOAD SAMPLE DATA
# ==============================

df_features = pd.DataFrame(list(db[FEATURE_COLLECTION].find().limit(50)))
df_clean = pd.DataFrame(list(db[CLEAN_COLLECTION].find().limit(50)))

if df_features.empty:
    raise Exception(f"{FEATURE_COLLECTION} is EMPTY!")

if df_clean.empty:
    raise Exception(f"{CLEAN_COLLECTION} is EMPTY!")

# Drop Mongo _id
df_features.drop(columns=["_id"], errors="ignore", inplace=True)
df_clean.drop(columns=["_id"], errors="ignore", inplace=True)

# ==============================
# COLUMN CHECK
# ==============================

features_cols = set(df_features.columns)
clean_cols = set(df_clean.columns)

missing_in_clean = features_cols - clean_cols
extra_in_clean = clean_cols - features_cols

print("Columns in aqi_features:", len(features_cols))
print("Columns in feature_cleaned:", len(clean_cols))

print("\n-----------------------------")

if not missing_in_clean and not extra_in_clean:
    print("✅ PERFECT — Columns match!")
else:
    print("⚠️ SCHEMA MISMATCH FOUND!\n")

    if missing_in_clean:
        print("❌ Missing in feature_cleaned:")
        for col in missing_in_clean:
            print("   ->", col)

    if extra_in_clean:
        print("\n⚠️ Extra columns in feature_cleaned:")
        for col in extra_in_clean:
            print("   ->", col)

print("\n-----------------------------")

# ==============================
# DTYPE CHECK
# ==============================

dtype_mismatch = []

common_cols = features_cols.intersection(clean_cols)

for col in common_cols:
    if df_features[col].dtype != df_clean[col].dtype:
        dtype_mismatch.append(
            (col, df_features[col].dtype, df_clean[col].dtype)
        )

if dtype_mismatch:
    print("⚠️ DTYPE MISMATCH:\n")
    for col, d1, d2 in dtype_mismatch:
        print(f"{col} -> aqi_features: {d1} | feature_cleaned: {d2}")
else:
    print("✅ Dtypes match.")

print("\n-----------------------------")

# ==============================
# TARGET COLUMN CHECK
# ==============================

TARGET = "target_aqi_t_plus_24h"   # change if needed

if TARGET in df_clean.columns:
    print(f"✅ Target column FOUND: {TARGET}")
else:
    print(f"❌ TARGET COLUMN MISSING: {TARGET}")
    print("🚨 Training will FAIL!")

print("\n========== FINAL VERDICT ==========\n")

if not missing_in_clean and not extra_in_clean and not dtype_mismatch and TARGET in df_clean.columns:
    print("🟢 SAFE TO SWITCH TRAINING → feature_cleaned")
else:
    print("🔴 DO NOT TRAIN YET — Fix schema first!")

print("\n===================================\n")
