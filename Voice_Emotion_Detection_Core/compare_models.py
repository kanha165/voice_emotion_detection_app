import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from utils.text_preprocessing import clean_text

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "processed", "train_processed.csv")
VAL_PATH   = os.path.join(BASE_DIR, "dataset", "processed", "val_processed.csv")

print(" Loading processed dataset...")

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)

# =========================
# DATA PREP
# =========================
X_train = train_df["text"].apply(clean_text)
y_train = train_df["emotion"]

X_val = val_df["text"].apply(clean_text)
y_val = val_df["emotion"]

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(max_features=15000)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec   = vectorizer.transform(X_val)

# =========================
# ALL CLASSIFICATION MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

# =========================
# TRAIN + EVALUATE LOOP
# =========================
results = []

print("\n MODEL COMPARISON RESULTS\n")

for name, model in models.items():
    print(f" Training {name}...")

    model.fit(X_train_vec, y_train)
    preds = model.predict(X_val_vec)

    acc = accuracy_score(y_val, preds)
    f1  = f1_score(y_val, preds, average="weighted")

    results.append((name, acc, f1))

    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-score: {f1:.4f}\n")

# =========================
# FINAL SUMMARY
# =========================
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "F1-score"]
).sort_values(by="Accuracy", ascending=False)

print("FINAL RANKING (Best → Worst)\n")
print(results_df.to_string(index=False))
