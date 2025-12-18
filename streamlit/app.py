# streamlit/app.py
import os
import requests
import streamlit as st

st.set_page_config(page_title="Cosmetics Review Classifier", layout="wide")

API_URL = os.getenv("API_URL", "https://ds-ml-final.onrender.com").rstrip("/")
PREDICT_URL = f"{API_URL}/predict"

st.title("Cosmetics Review Classifier")
st.write("Demo 1: **Text-dominant sentiment** • Demo 2: **Rating-dominant score** • Optional: **Hybrid blend**")

with st.sidebar:
    st.subheader("API")
    st.write(f"API URL used: {API_URL}")
    st.write(f"Predict endpoint: {PREDICT_URL}")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    review_text = st.text_area("Review text", value="Amazing moisturizer", height=180)
    rating = st.slider("Rating", 1, 5, 5)  # set default 5 for demo positivity
    helpful_votes = st.number_input("Helpful votes", min_value=0, value=0, step=1)
    verified_purchase = st.checkbox("Verified purchase", value=True)

with col2:
    st.subheader("Demo mode")
    mode = st.radio(
        "Choose behavior",
        options=["text", "rating", "hybrid"],
        format_func=lambda x: {
            "text": "Text-dominant (sentiment)",
            "rating": "Rating-dominant (review score)",
            "hybrid": "Hybrid (blend)",
        }[x],
        index=0
    )

    text_weight = 0.8
    if mode == "hybrid":
        text_weight = st.slider("Text weight (hybrid)", 0.0, 1.0, 0.8, 0.05)

    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)

predict_btn = st.button("Predict", type="primary")

if predict_btn:
    payload = {
        "review_text": review_text,
        "rating": rating,
        "helpful_votes": helpful_votes,
        "verified_purchase": verified_purchase,
        "mode": mode,
        "text_weight": text_weight,
        "threshold": threshold,
    }

    try:
        r = requests.post(PREDICT_URL, json=payload, timeout=60)
        r.raise_for_status()
        out = r.json()

        pred = int(out.get("prediction", out.get("label", 0)))

        # accept either key
        prob = out.get("probability", out.get("probability_positive", 0.0))
        prob = float(prob)

        label_txt = "positive (label=1)" if pred == 1 else "negative (label=0)"
        if pred == 1:
            st.success(f"Prediction: **{label_txt}**  •  p_positive={prob:.3f}")
        else:
            st.error(f"Prediction: **{label_txt}**  •  p_positive={prob:.3f}")

        st.subheader("Explanation")
        exp = out.get("explain", {})
        st.write(
            f"- mode: **{out.get('mode', mode)}**\n"
            f"- p_text: **{float(exp.get('p_text', 0.0)):.3f}**\n"
            f"- p_rating: **{float(exp.get('p_rating', 0.0)):.3f}**\n"
            f"- text_weight used: **{float(out.get('text_weight', 0.0)):.2f}**\n"
            f"- threshold: **{float(out.get('threshold', threshold)):.2f}**\n"
            f"- p_final: **{float(exp.get('p_final', prob)):.3f}**"
        )

        with st.expander("Raw API response"):
            st.json(out)

    except Exception as e:
        st.error(f"API call failed: {e}")
        st.write(f"Tried URL: {PREDICT_URL}")
