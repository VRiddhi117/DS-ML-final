import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.joblib")  # default inside /app

# -----------------------------
# Load model at startup
# -----------------------------
model = joblib.load(MODEL_PATH)

# -----------------------------
# API
# -----------------------------
app = FastAPI(title="Cosmetics Review Classifier", version="0.1.0")

class ReviewInput(BaseModel):
    review_text: str
    rating: int
    helpful_votes: int
    verified_purchase: Union[bool, int]

@app.get("/")
def home():
    return {"status": "ok", "message": "Cosmetics Review Classifier API running"}

@app.post("/predict")
def predict(inp: ReviewInput):
    # Normalize verified_purchase into 0/1
    verified = inp.verified_purchase
    if isinstance(verified, bool):
        verified = int(verified)
    else:
        verified = int(verified)

    X = pd.DataFrame([{
        "review_text": inp.review_text,
        "rating": inp.rating,
        "helpful_votes": inp.helpful_votes,
        "verified_purchase": verified
    }])

    pred = int(model.predict(X)[0])

    # If your pipeline supports predict_proba, return probability too
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])

    return {
        "prediction": pred,
        "label": pred,
        "probability_positive": prob
    }
