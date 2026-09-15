# compare_classifiers.py
# Compares a few linear classifiers on the same TF-IDF features.
# Run: python src/models/compare_classifiers.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

DATA_PATH = "data/processed/complaints_clean.csv"
SEED = 42
TEST_SIZE = 0.2

print(f"loading {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"rows: {len(df):,}")

X = df["clean_text"].astype(str)
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"vocab: {len(vectorizer.vocabulary_):,}\n")

models = {
    "LogReg": LogisticRegression(
        max_iter=1000, class_weight="balanced", C=1.0, random_state=SEED
    ),
    "LinearSVC": LinearSVC(
        class_weight="balanced", C=1.0, random_state=SEED, max_iter=3000
    ),
    "SGD": SGDClassifier(
        loss="modified_huber",
        class_weight="balanced",
        random_state=SEED,
        max_iter=2000,
    ),
}

results = {}
for name, model in models.items():
    print(f"training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    results[name] = {"accuracy": acc, "macro_f1": macro_f1}
    print(f"  accuracy: {acc:.4f}  macro_f1: {macro_f1:.4f}")

print("\nsummary:")
summary = pd.DataFrame(results).T.sort_values("macro_f1", ascending=False)
print(summary)