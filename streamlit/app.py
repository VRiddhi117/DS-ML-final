import os
import time
import requests
import streamlit as st

st.set_page_config(page_title="Cosmetics Review Classifier", layout="wide")

# Use Render by default; allow override with env var
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
    rating = st.slider("Rating", 1, 5, 2)
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
        index=2
    )

    text_weight = 0.8
    if mode == "hybrid":
        text_weight = st.slider("Text weight (hybrid)", 0.0, 1.0, 0.8, 0.05)

    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)

predict_btn = st.button("Predict", type="primary")

def post_with_retries(url, payload, tries=3, timeout=90):
    last_err = None
    for i in range(tries):
        try:
            return requests.post(url, json=payload, timeout=timeout)
        except Exception as e:
            last_err = e
            time.sleep(2 * (i + 1))  # small backoff
    raise last_err

if predict_btn:
    payload = {
        "review_text": review_text,
        "rating": int(rating),
        "helpful_votes": int(helpful_votes),
        "verified_purchase": bool(verified_purchase),
        "mode": mode,
        "text_weight": float(text_weight),
        "threshold": float(threshold),
    }

    try:
        # Render cold-start fix: longer timeout + retries
        r = post_with_retries(PREDICT_URL, payload, tries=3, timeout=90)

        # If 422, show exact FastAPI validation error (super helpful)
        if r.status_code == 422:
            st.error("422 Validation Error (your request JSON doesn’t match the API model).")
            st.json(r.json())
            st.stop()

        r.raise_for_status()
        out = r.json()

        # ✅ These keys match the FastAPI code I gave you
        pred = out.get("prediction", out.get("label"))
        prob = out.get("probability_positive")

        if prob is None:
            st.error("API response missing 'probability_positive'. Raw response below:")
            st.json(out)
            st.stop()

        label_txt = "positive (label=1)" if int(pred) == 1 else "negative (label=0)"
        if int(pred) == 1:
            st.success(f"Prediction: **{label_txt}**  •  p_positive={prob:.3f}")
        else:
            st.error(f"Prediction: **{label_txt}**  •  p_positive={prob:.3f}")

        st.subheader("Explanation")
        exp = out.get("explain", {})
        st.write(
            f"- mode: **{out.get('mode')}**\n"
            f"- p_text: **{exp.get('p_text', 0):.3f}**\n"
            f"- p_rating: **{exp.get('p_rating', 0):.3f}**\n"
            f"- text_weight used: **{out.get('text_weight', 0):.2f}**\n"
            f"- threshold: **{out.get('threshold', 0):.2f}**\n"
            f"- p_final: **{exp.get('p_final', 0):.3f}**"
        )

        with st.expander("Raw API response"):
            st.json(out)

    except Exception as e:
        st.error(f"API call failed: {e}")
        st.write(f"Tried URL: {PREDICT_URL}")
