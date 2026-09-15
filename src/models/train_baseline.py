"""
Train a TF-IDF + Logistic Regression baseline for complaint classification.

Usage:
    python src/models/train_baseline.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ------- Config -------
DATA_PATH = "data/processed/complaints_clean.csv"
MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "logreg_baseline.pkl")

SEED = 42
TEST_SIZE = 0.2   # 20% held out for test

# ------- Load data -------
print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Rows: {len(df):,}")
print(f"Classes: {df['category'].value_counts().to_dict()}")

X = df["clean_text"].astype(str)
y = df["category"]

# ------- Split -------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

# ------- Vectorize -------
print("\nFitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")

# ------- Train -------
print("\nTraining Logistic Regression...")
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=1.0,
    random_state=SEED,
)
model.fit(X_train_tfidf, y_train)

# ------- Evaluate -------
print("\nEvaluating on test set...")
y_pred = model.predict(X_test_tfidf)

print("\n" + "=" * 60)
print("Classification Report:")
print("=" * 60)
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------- Save -------
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(model, MODEL_PATH)
print(f"\nSaved vectorizer: {VECTORIZER_PATH}")
print(f"Saved model: {MODEL_PATH}")