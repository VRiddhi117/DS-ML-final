from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import numpy as np

app = FastAPI(title="Cosmetics Review Sentiment API")

# Load model once at startup
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "best_model.joblib"))
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    review_text: str


@app.get("/")
def root():
    return {"status": "ok", "message": "Cosmetics sentiment API is running"}


@app.post("/predict")
def predict(req: PredictRequest):
    text = (req.review_text or "").strip()

    if not text:
        return {
            "prediction": 0,
            "label": 0,
            "probability_positive": 0.0,
            "detail": "Empty review_text"
        }

    # Most sklearn text pipelines accept list[str]
    X = [text]

    prob_pos = None

    # 1) Preferred: predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # probability of class 1 (positive)
        prob_pos = float(proba[0][1])

    # 2) Fallback: decision_function -> sigmoid
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        prob_pos = float(1.0 / (1.0 + np.exp(-score)))

    # 3) Fallback: predict only (no probability available)
    else:
        pred = int(model.predict(X)[0])
        prob_pos = 1.0 if pred == 1 else 0.0

    pred = 1 if prob_pos >= 0.5 else 0

    return {
        "prediction": pred,
        "label": pred,
        "probability_positive": prob_pos
    }
