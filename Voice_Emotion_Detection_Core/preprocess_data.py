import pandas as pd
import os
from utils.text_preprocessing import clean_text

# =========================
# PATHS
# =========================
DATASET_DIR = "dataset"
OUTPUT_DIR = "dataset/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_file(filename):
    print(f"Processing {filename}...")

    # IMPORTANT: separator is ;
    df = pd.read_csv(os.path.join(DATASET_DIR, filename), sep=";")

    # Columns rename (clean design)
    df.columns = ["text", "emotion"]

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Optional: normalize emotion names
    df["emotion"] = df["emotion"].str.lower().str.strip()

    return df

# =========================
# PROCESS FILES
# =========================
train_df = preprocess_file("train.csv")
val_df   = preprocess_file("validation.csv")
test_df  = preprocess_file("test.csv")

# =========================
# SAVE OUTPUT
# =========================
train_df.to_csv(os.path.join(OUTPUT_DIR, "train_processed.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val_processed.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_processed.csv"), index=False)

print("\n PREPROCESSING COMPLETED SUCCESSFULLY")
print(" Processed files saved in dataset/processed/")
