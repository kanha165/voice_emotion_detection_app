import os
import pickle   
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.text_preprocessing import clean_text


# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_PATH = os.path.join(
    BASE_DIR, "dataset", "processed", "test_processed.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "model")


# =========================
# LOAD DATA
# =========================
print(" Loading test data...")

test_df = pd.read_csv(TEST_PATH)

X_test = test_df["text"].apply(clean_text)
y_test = test_df["emotion"]


# =========================
# LOAD MODEL
# =========================
vectorizer = pickle.load(
    open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb")
)
model = pickle.load(
    open(os.path.join(MODEL_DIR, "linear_svm.pkl"), "rb")
)

X_test_vec = vectorizer.transform(X_test)


# =========================
# EVALUATION
# =========================
print("\n FINAL TEST SET RESULTS")

preds = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, preds)

print(f"Accuracy: {accuracy:.4f}\n")

print("Classification Report:\n")
print(classification_report(y_test, preds))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, preds))
