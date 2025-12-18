from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(title="Cosmetics Review Classifier")

# Load model from same folder as this app.py
MODEL_PATH = Path(__file__).parent / "best_model.joblib"
model = joblib.load(MODEL_PATH)

class ReviewInput(BaseModel):
    review_text: str
    rating: int
    helpful_votes: int
    verified_purchase: Union[bool, int]

@app.get("/health")
def health():
    return {"status": "ok", "model_path": str(MODEL_PATH)}

@app.post("/predict")
def predict(inp: ReviewInput):
    # normalize verified_purchase
    verified = int(bool(inp.verified_purchase))

    X = pd.DataFrame([{
        "review_text": inp.review_text,
        "rating": int(inp.rating),
        "helpful_votes": int(inp.helpful_votes),
        "verified_purchase": verified
    }])

    # predict label
    y_pred = int(model.predict(X)[0])

    # probability if available
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])

    return {
        "label": y_pred,
        "sentiment": "positive" if y_pred == 1 else "negative",
        "prob_positive": proba
    }
