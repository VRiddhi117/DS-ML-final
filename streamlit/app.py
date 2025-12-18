import os
import requests
import streamlit as st

st.set_page_config(page_title="Cosmetics Review Classifier", layout="wide")
st.title("Cosmetics Review Classifier")
st.write("Predict whether a review is **positive (1)** or **negative (0)**")

# inside docker-compose use http://api:8000
# local run use http://localhost:8000
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

review_text = st.text_area("Review text", value="", height=150)
rating = st.slider("Rating", 1, 5, 3)
helpful_votes = st.number_input("Helpful votes", min_value=0, value=0, step=1)
verified_purchase = st.checkbox("Verified purchase", value=False)

if st.button("Predict"):
    payload = {
        "review_text": review_text,
        "rating": int(rating),
        "helpful_votes": int(helpful_votes),
        "verified_purchase": bool(verified_purchase),
    }

    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        out = r.json()

        label = out["label"]
        sentiment = out["sentiment"]
        proba = out.get("prob_positive", None)

        if label == 1:
            st.success(f"Prediction: {sentiment} (label=1)")
        else:
            st.error(f"Prediction: {sentiment} (label=0)")

        if proba is not None:
            st.write(f"Prob(positive): **{proba:.3f}**")

        st.caption(f"API URL used: {API_URL}")

    except Exception as e:
        st.error(f"API call failed: {e}")
        st.caption(f"Tried URL: {API_URL}/predict")
