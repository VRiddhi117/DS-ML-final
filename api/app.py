# api/app.py
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

app = FastAPI(title="Cosmetics Review API", version="1.0")

MODEL_PATH = Path(__file__).parent / "best_model.joblib"
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    review_text: str = Field(..., min_length=1)
    rating: int | None = None
    helpful_votes: int | None = None
    verified_purchase: bool | None = None

    # demo switches (optional)
    mode: str | None = "text"          # "text" | "rating" | "hybrid"
    text_weight: float | None = 0.8
    threshold: float | None = 0.5


@app.get("/")
def root():
    return {"status": "ok", "message": "Cosmetics Review API running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


def rating_to_prob(rating: int | None) -> float:
    if rating is None:
        return 0.5
    mapping = {1: 0.05, 2: 0.15, 3: 0.35, 4: 0.75, 5: 0.90}
    return float(mapping.get(int(rating), 0.5))


@app.post("/predict")
def predict(req: PredictRequest):
    text = req.review_text.strip()

    # TEXT probability (your TFIDF + LogisticRegression pipeline)
    try:
        p_text = float(model.predict_proba([text])[0][1])
    except Exception:
        # fallback if model doesn't support predict_proba (rare)
        pred = int(model.predict([text])[0])
        p_text = 1.0 if pred == 1 else 0.0

    # RATING probability (simple rule-map)
    p_rating = rating_to_prob(req.rating)

    mode = (req.mode or "text").lower()
    text_weight = float(req.text_weight if req.text_weight is not None else 0.8)
    threshold = float(req.threshold if req.threshold is not None else 0.5)

    if mode == "rating":
        p_final = p_rating
        used_tw = 0.0
    elif mode == "hybrid":
        p_final = text_weight * p_text + (1.0 - text_weight) * p_rating
        used_tw = text_weight
    else:
        # default text-dominant
        p_final = p_text
        used_tw = 1.0

    prediction = 1 if p_final >= threshold else 0

    return {
        "prediction": prediction,
        "label": prediction,
        "probability": p_final,
        "probability_positive": p_final,  # keeps compatibility if you used this earlier
        "mode": mode,
        "text_weight": used_tw,
        "threshold": threshold,
        "explain": {
            "p_text": p_text,
            "p_rating": p_rating,
            "p_final": p_final
        }
    }
