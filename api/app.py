from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("best_model.joblib")

class PredictRequest(BaseModel):
    review_text: str
    rating: int
    helpful_votes: int
    verified_purchase: bool
    mode: str = "text"       # text | rating | hybrid
    text_weight: float = 0.8
    threshold: float = 0.5


@app.post("/predict")
def predict(req: PredictRequest):
    # 1️⃣ Text probability
    p_text = float(model.text_model.predict_proba([req.review_text])[0, 1])

    # 2️⃣ Rating probability (rule-based)
    rating_map = {1: 0.05, 2: 0.15, 3: 0.35, 4: 0.75, 5: 0.90}
    p_rating = rating_map.get(req.rating, 0.5)

    # 3️⃣ Combine based on mode
    if req.mode == "text":
        p_final = p_text
        text_weight = 1.0
    elif req.mode == "rating":
        p_final = p_rating
        text_weight = 0.0
    else:  # hybrid
        text_weight = req.text_weight
        p_final = text_weight * p_text + (1 - text_weight) * p_rating

    label = int(p_final >= req.threshold)

    return {
        "prediction": label,
        "label": label,
        "probability_positive": p_final,
        "mode": req.mode,
        "text_weight": text_weight,
        "threshold": req.threshold,
        "explain": {
            "p_text": p_text,
            "p_rating": p_rating,
            "p_final": p_final
        }
    }
