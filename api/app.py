# api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import re

app = FastAPI()

TEXT_MODEL_PATH = "models/text_model.joblib"
BEST_MODEL_PATH = "models/best_model.joblib"

text_model = None
best_model = None

# Load models safely
try:
    text_model = joblib.load(TEXT_MODEL_PATH)
except Exception:
    text_model = None

try:
    best_model = joblib.load(BEST_MODEL_PATH)
except Exception:
    best_model = None


class PredictRequest(BaseModel):
    review_text: str
    rating: int
    helpful_votes: int = 0
    verified_purchase: bool = False
    verified_boost: float = 0.0
    mode: str = "text"  # text | rating | hybrid
    text_weight: float = 0.8
    threshold: float = 0.5


NEGATION_PATTERNS = [
    r"\bnot good\b",
    r"\bnot great\b",
    r"\bnot worth\b",
    r"\bnot happy\b",
    r"\bno good\b",
    r"\bnever again\b",
]


def contains_negation(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in NEGATION_PATTERNS)


@app.get("/")
def health():
    return {
        "status": "ok",
        "text_model_loaded": text_model is not None,
        "best_model_loaded": best_model is not None,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    text = req.review_text.strip()
    mode = req.mode

    # ---------- TEXT PROB ----------
    if text_model:
        p_text = float(text_model.predict_proba([text])[0][1])
    else:
        p_text = 0.5

    # 🔥 NEGATION FIX (CRITICAL)
    if contains_negation(text):
        p_text = min(p_text, 0.25)

    # ---------- RATING PROB ----------
    if req.rating >= 4:
        p_rating = 0.95
    elif req.rating <= 2:
        p_rating = 0.05
    else:
        p_rating = 0.5

    # ---------- FINAL PROB ----------
    if mode == "text":
        p_final = p_text
    elif mode == "rating":
        p_final = p_rating
    else:
        p_final = req.text_weight * p_text + (1 - req.text_weight) * p_rating

    # Verified purchase boost
    if req.verified_purchase:
        p_final = min(1.0, p_final + req.verified_boost)

    prediction = int(p_final >= req.threshold)

    return {
        "prediction": prediction,
        "probability_positive": p_final,
        "mode": mode,
        "threshold": req.threshold,
        "explain": {
            "p_text": p_text,
            "p_rating": p_rating,
            "p_final": p_final,
            "negation_detected": contains_negation(text),
            "verified_purchase": req.verified_purchase,
        },
    }
