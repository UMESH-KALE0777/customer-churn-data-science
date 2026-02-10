import pandas as pd
import os

# Paths
RAW_PATH = "data/raw/customer_churn_raw.csv"
PROCESSED_PATH = "data/processed/churn_clean.csv"

# Create processed folder if not exists
os.makedirs("data/processed", exist_ok=True)

# 1️⃣ Load raw data
df = pd.read_csv(RAW_PATH)

# 2️⃣ Normalize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# 3️⃣ Convert churn label to binary
df["churn_label"] = df["churn_label"].map({"Yes": 1, "No": 0})

# 4️⃣ Drop ID column
if "customer_id" in df.columns:
    df = df.drop(columns=["customer_id"])

# 5️⃣ Handle missing values (simple & safe)
df = df.dropna()

# 6️⃣ Save cleaned data ✅
df.to_csv(PROCESSED_PATH, index=False)

print("✅ churn_clean.csv created successfully")
print("Shape:", df.shape)
