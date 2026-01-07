import os
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from utils.text_preprocessing import clean_text


# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(
    BASE_DIR, "dataset", "processed", "train_processed.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================
print(" Loading training data...")

train_df = pd.read_csv(TRAIN_PATH)

X_train = train_df["text"].apply(clean_text)
y_train = train_df["emotion"]


# =========================
# VECTORIZE
# =========================
print(" Vectorizing text (TF-IDF)...")

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)


# =========================
# TRAIN FINAL MODEL
# =========================
print(" Training Linear SVM...")

model = LinearSVC()
model.fit(X_train_vec, y_train)


# =========================
# SAVE MODEL
# =========================
pickle.dump(model, open(os.path.join(MODEL_DIR, "linear_svm.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb"))

print(" Final Linear SVM model & vectorizer saved successfully")
