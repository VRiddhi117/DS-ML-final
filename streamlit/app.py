import os
import requests
import streamlit as st

st.set_page_config(page_title="Cosmetics Review Classifier", layout="wide")

# Use deployed API by default, but allow override via env var
API_URL = os.getenv("API_URL", "https://ds-ml-final.onrender.com").rstrip("/")
PREDICT_URL = f"{API_URL}/predict"

st.title("Cosmetics Review Classifier (Text-only)")
st.write("Predict whether a cosmetics review is **positive (1)** or **negative (0)** using the review text.")

with st.sidebar:
    st.subheader("API")
    st.write(f"API URL used: {API_URL}")
    st.write(f"Predict endpoint: {PREDICT_URL}")

review_text = st.text_area("Review text", value="Amazing moisturizer", height=180)
predict_btn = st.button("Predict", type="primary")

if predict_btn:
    payload = {"review_text": review_text}

    try:
        r = requests.post(PREDICT_URL, json=payload, timeout=60)
        r.raise_for_status()
        out = r.json()

        pred = int(out.get("prediction", out.get("label", 0)))
        prob = float(out.get("probability_positive", 0.0))

        label_txt = "positive (label=1)" if pred == 1 else "negative (label=0)"

        if pred == 1:
            st.success(f"Prediction: **{label_txt}**  •  p_positive={prob:.3f}")
        else:
            st.error(f"Prediction: **{label_txt}**  •  p_positive={prob:.3f}")

        with st.expander("Raw API response"):
            st.json(out)

    except Exception as e:
        st.error(f"API call failed: {e}")
        st.write(f"Tried URL: {PREDICT_URL}")
        st.info("If you're using Render free tier, first request may be slow. Try again once it wakes up.")
