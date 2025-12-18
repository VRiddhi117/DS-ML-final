# api/app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from joblib import load


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Cosmetics Review Classifier API", version="1.0")


# ----------------------------
# Paths / Models
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

TEXT_MODEL_PATH = Path(os.getenv("TEXT_MODEL_PATH", str(REPO_ROOT / "models" / "text_model.joblib")))
BEST_MODEL_PATH = Path(os.getenv("BEST_MODEL_PATH", str(REPO_ROOT / "models" / "best_model.joblib")))

TEXT_MODEL = None
BEST_MODEL = None


@app.on_event("startup")
def load_models() -> None:
    global TEXT_MODEL, BEST_MODEL

    if TEXT_MODEL_PATH.exists():
        TEXT_MODEL = load(TEXT_MODEL_PATH)
        print(f"[OK] Loaded TEXT_MODEL from: {TEXT_MODEL_PATH}")
    else:
        TEXT_MODEL = None
        print(f"[WARN] TEXT_MODEL not found at: {TEXT_MODEL_PATH}")

    if BEST_MODEL_PATH.exists():
        BEST_MODEL = load(BEST_MODEL_PATH)
        print(f"[OK] Loaded BEST_MODEL from: {BEST_MODEL_PATH}")
    else:
        BEST_MODEL = None
        print(f"[WARN] BEST_MODEL not found at: {BEST_MODEL_PATH}")


# ----------------------------
# Request/Response Schemas
# ----------------------------
class PredictRequest(BaseModel):
    review_text: str = Field(..., min_length=1)
    rating: int = Field(3, ge=1, le=5)
    helpful_votes: int = Field(0, ge=0)
    verified_purchase: bool = True

    mode: Literal["text", "rating", "hybrid"] = "hybrid"
    text_weight: float = Field(0.80, ge=0.0, le=1.0)  # used only for fallback blend
    threshold: float = Field(0.50, ge=0.0, le=1.0)

    # Optional transparent post-adjust (kept optional so old clients won't break)
    verified_boost: float = Field(0.0, ge=0.0, le=0.15)


class PredictResponse(BaseModel):
    prediction: int
    probability_positive: float
    mode: str
    text_weight: float
    threshold: float
    explain: Dict[str, Any]


# ----------------------------
# Utilities
# ----------------------------
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def rating_to_prob(rating: int) -> float:
    # Simple interpretable baseline
    mapping = {1: 0.05, 2: 0.20, 3: 0.50, 4: 0.80, 5: 0.95}
    return float(mapping.get(int(rating), 0.50))


def text_to_prob(review_text: str) -> float:
    # If trained text model exists, use it
    if TEXT_MODEL is not None:
        try:
            p = float(TEXT_MODEL.predict_proba([review_text])[0][1])
            return clamp01(p)
        except Exception as e:
            print(f"[WARN] TEXT_MODEL predict_proba failed: {e}")

    # Fallback heuristic (only if model missing/broken)
    t = (review_text or "").lower()
    if any(w in t for w in ["amazing", "great", "love", "perfect", "excellent", "awesome"]):
        return 0.80
    if any(w in t for w in ["terrible", "awful", "hate", "bad", "worst", "horrible"]):
        return 0.20
    return 0.50


def best_model_prob(req: PredictRequest) -> Dict[str, Any]:
    """
    Try to get p_positive from BEST_MODEL.

    Returns dict:
      {
        "ok": bool,
        "p_best": float|nan,
        "error": str|None,
        "input_variant": "dataframe"|"records"|None
      }
    """
    if BEST_MODEL is None:
        return {"ok": False, "p_best": float("nan"), "error": "BEST_MODEL not loaded", "input_variant": None}

    # Common feature names across projects (we try a few variants)
    row = {
        "review_text": req.review_text,
        "text": req.review_text,
        "rating": req.rating,
        "helpful_votes": req.helpful_votes,
        "verified_purchase": req.verified_purchase,
        "verified": req.verified_purchase,
    }

    # Variant A: DataFrame with the most likely column names
    df = pd.DataFrame(
        [{
            "review_text": req.review_text,
            "rating": req.rating,
            "helpful_votes": req.helpful_votes,
            "verified_purchase": req.verified_purchase,
        }]
    )

    try:
        if hasattr(BEST_MODEL, "predict_proba"):
            p = float(BEST_MODEL.predict_proba(df)[0][1])
        else:
            # If it's a decision-function model, approximate via sigmoid
            scores = BEST_MODEL.decision_function(df)
            p = float(1 / (1 + np.exp(-scores[0])))
        return {"ok": True, "p_best": clamp01(p), "error": None, "input_variant": "dataframe"}
    except Exception as e_df:
        # Variant B: list of dict records (some pipelines prefer this)
        try:
            records = [{
                "review_text": req.review_text,
                "rating": req.rating,
                "helpful_votes": req.helpful_votes,
                "verified_purchase": req.verified_purchase,
            }]
            if hasattr(BEST_MODEL, "predict_proba"):
                p = float(BEST_MODEL.predict_proba(records)[0][1])
            else:
                scores = BEST_MODEL.decision_function(records)
                p = float(1 / (1 + np.exp(-scores[0])))
            return {"ok": True, "p_best": clamp01(p), "error": None, "input_variant": "records"}
        except Exception as e_rec:
            return {
                "ok": False,
                "p_best": float("nan"),
                "error": f"BEST_MODEL inference failed. dataframe_err={e_df} | records_err={e_rec}",
                "input_variant": None,
            }


def apply_verified_boost(p: float, verified: bool, boost: float) -> (float, float):
    """
    Transparent post-adjustment (optional).
    Returns (p_new, boost_applied).
    """
    if verified and boost and boost > 0:
        p2 = clamp01(p + float(boost))
        return p2, float(boost)
    return p, 0.0


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "text_model_loaded": TEXT_MODEL is not None,
        "best_model_loaded": BEST_MODEL is not None,
        "text_model_path": str(TEXT_MODEL_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # Always compute these for transparency
    p_text = text_to_prob(req.review_text)
    p_rating = rating_to_prob(req.rating)

    # HYBRID uses BEST_MODEL (option 2)
    best_info = best_model_prob(req)
    p_best = float(best_info.get("p_best", float("nan")))

    # Enforce mode strictly
    used_w = float(req.text_weight)  # only used in fallback blend explain
    p_final: float

    if req.mode == "text":
        p_final = p_text
    elif req.mode == "rating":
        p_final = p_rating
    else:
        # hybrid
        if best_info["ok"] and (p_best == p_best):  # not NaN
            p_final = p_best
        else:
            # fallback blend so your API still works even if best_model breaks
            w = clamp01(float(req.text_weight))
            used_w = w
            p_final = w * p_text + (1.0 - w) * p_rating

    # Optional verified boost
    p_final, boost_applied = apply_verified_boost(p_final, req.verified_purchase, req.verified_boost)

    # Threshold decision
    th = clamp01(float(req.threshold))
    pred = int(p_final >= th)

    explain: Dict[str, Any] = {
        "p_text": float(p_text),
        "p_rating": float(p_rating),
        "p_best_model": float(p_best) if (p_best == p_best) else None,
        "best_model_ok": bool(best_info["ok"]),
        "best_model_input_variant": best_info.get("input_variant"),
        "best_model_error": best_info.get("error") if not best_info["ok"] else None,
        "p_final": float(p_final),
        "boost_applied": float(boost_applied),
        "verified_purchase": bool(req.verified_purchase),
        "note": (
            "Hybrid uses BEST_MODEL when available; falls back to (text_weight*p_text + (1-text_weight)*p_rating) if BEST_MODEL fails."
        ),
    }

    return PredictResponse(
        prediction=int(pred),
        probability_positive=float(p_final),
        mode=req.mode,
        text_weight=float(used_w),
        threshold=float(th),
        explain=explain,
    )
