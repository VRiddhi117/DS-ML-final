import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

app = FastAPI(title="Cosmetics Review Classifier")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/best_model.joblib")
model = joblib.load(MODEL_PATH)

class ReviewInput(BaseModel):
    review_text: str
    rating: int
    helpful_votes: int = 0
    verified_purchase: bool = False

    # demo controls
    mode: Literal["text", "rating", "hybrid"] = "text"
    text_weight: float = 0.8
    threshold: float = 0.5


def rating_to_prob(rating: int) -> float:
    # simple monotonic mapping (works well for demo)
    mapping = {1: 0.05, 2: 0.15, 3: 0.35, 4: 0.75, 5: 0.90}
    return float(mapping.get(int(rating), 0.50))


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict(inp: ReviewInput):
    # Text model probability
    # If your saved model is a sklearn Pipeline (TfidfVectorizer + LogisticRegression),
    # it will support predict_proba on text directly.
    p_text = float(model.predict_proba([inp.review_text])[0][1])

    # Rating probability
    p_rating = rating_to_prob(inp.rating)

    # Decide final prob based on mode
    if inp.mode == "text":
        p_final = p_text
        used_weight = 1.0
    elif inp.mode == "rating":
        p_final = p_rating
        used_weight = 0.0
    else:
        w = min(max(inp.text_weight, 0.0), 1.0)
        p_final = w * p_text + (1.0 - w) * p_rating
        used_weight = w

    pred = 1 if p_final >= inp.threshold else 0

    return {
        "prediction": pred,
        "label": pred,
        "probability": p_final,
        "probability_positive": p_final,
        "mode": inp.mode,
        "text_weight": used_weight,
        "threshold": inp.threshold,
        "explain": {
            "p_text": p_text,
            "p_rating": p_rating,
            "p_final": p_final
        }
    }
