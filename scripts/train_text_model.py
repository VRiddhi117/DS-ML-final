# scripts/train_text_model.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Always resolve paths from repo root (works no matter where you run from)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Default data file: first CSV found under data/raw
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_CSV = next(DEFAULT_RAW_DIR.glob("*.csv"), None)

DATA_PATH = Path(os.getenv("TRAIN_DATA", str(DEFAULT_CSV or (REPO_ROOT / "data" / "reviews.csv"))))
OUT_PATH = Path(os.getenv("TEXT_MODEL_OUT", str(REPO_ROOT / "models" / "text_model.joblib")))

# Try common column names in your cosmetics review dataset
TEXT_COL_CANDIDATES = ["review_text", "review", "text", "content", "Review Text"]
LABEL_COL_CANDIDATES = ["label", "sentiment", "target", "is_positive", "class"]
RATING_COL_CANDIDATES = ["rating", "score", "stars", "Rating"]

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}\n"
            f"Tip: set TRAIN_DATA env var or place csv in {DEFAULT_RAW_DIR}"
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    text_col = pick_col(df, TEXT_COL_CANDIDATES)
    label_col = pick_col(df, LABEL_COL_CANDIDATES)
    rating_col = pick_col(df, RATING_COL_CANDIDATES)

    if text_col is None:
        raise ValueError(
            f"Could not find a text column. Columns found:\n{list(df.columns)}\n"
            f"Expected one of: {TEXT_COL_CANDIDATES}"
        )

    # If dataset doesn't already have a label column, derive from rating as a fallback
    if label_col is None:
        if rating_col is None:
            raise ValueError(
                "No label column AND no rating column found to derive label.\n"
                f"Columns found: {list(df.columns)}"
            )
        # Derive label: >=4 positive, <=2 negative, drop neutral 3s (common approach)
        df = df.dropna(subset=[text_col, rating_col]).copy()
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
        df = df[df[rating_col].isin([1, 2, 4, 5])].copy()
        df["label"] = (df[rating_col] >= 4).astype(int)
        label_col = "label"
        print(f"[INFO] No label column found. Derived label from `{rating_col}` (>=4 positive, <=2 negative).")
    else:
        df = df.dropna(subset=[text_col, label_col]).copy()
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
        df = df.dropna(subset=[label_col]).copy()
        df[label_col] = df[label_col].astype(int)

    X = df[text_col].astype(str)
    y = df[label_col].astype(int)

    # Basic safety: need both classes
    if y.nunique() < 2:
        raise ValueError(f"Only one class found in labels. Unique labels: {sorted(y.unique().tolist())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )),
        ]
    )

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n===== EVAL =====")
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, pred, digits=4))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, OUT_PATH)
    print(f"\n✅ Saved text model to: {OUT_PATH}")

if __name__ == "__main__":
    main()
