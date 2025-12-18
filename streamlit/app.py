# streamlit/app.py
import os
import time
import requests
import streamlit as st

st.set_page_config(page_title="Cosmetics Review Classifier (Debug)", layout="wide")

DEFAULT_API_URL = os.getenv("API_URL", "https://ds-ml-final.onrender.com").rstrip("/")

st.title("Cosmetics Review Classifier")
st.caption("Debug build: shows exactly what Streamlit receives from the API.")

# ----------------------------
# Sidebar: choose API URL
# ----------------------------
with st.sidebar:
    st.subheader("API Settings")

    api_url = st.text_input(
        "API base URL",
        value=st.session_state.get("api_url", DEFAULT_API_URL),
        help="Use local: http://127.0.0.1:8000  |  Use Render: https://xxxx.onrender.com",
    ).strip().rstrip("/")

    st.session_state["api_url"] = api_url
    predict_url = f"{api_url}/predict"

    st.write("Using:")
    st.code(api_url)
    st.write("Predict endpoint:")
    st.code(predict_url)

    if st.button("Health check (GET /)", use_container_width=True):
        try:
            r = requests.get(api_url, timeout=10)
            st.success(f"GET / → {r.status_code}")
            try:
                st.json(r.json())
            except Exception:
                st.code(r.text)
        except Exception as e:
            st.error(f"GET / failed: {e}")

    st.divider()
    show_payload = st.checkbox("Show payload", value=True)
    show_response = st.checkbox("Show response", value=True)

# ----------------------------
# Inputs
# ----------------------------
c1, c2 = st.columns([2, 1], gap="large")

with c1:
    st.subheader("Review inputs")
    review_text = st.text_area("Review text", value="Amazing moisturizer", height=140)
    rating = st.slider("Rating (1–5)", 1, 5, 2)
    helpful_votes = st.number_input("Helpful votes", min_value=0, value=0, step=1)
    verified_purchase = st.checkbox("Verified purchase", value=True)

with c2:
    st.subheader("Mode")
    mode = st.radio(
        "Prediction mode",
        ["text", "rating", "hybrid"],
        index=0,
        format_func=lambda x: {
            "text": "Text-only",
            "rating": "Rating-only",
            "hybrid": "Hybrid",
        }[x],
    )

    if mode == "hybrid":
        text_weight = st.slider("text_weight", 0.0, 1.0, 0.80, 0.05)
    elif mode == "text":
        text_weight = 1.0
        st.caption("Text-only ignores rating.")
    else:
        text_weight = 0.0
        st.caption("Rating-only ignores text.")

    threshold = st.slider("threshold", 0.0, 1.0, 0.50, 0.05)
    verified_boost = st.slider("verified_boost", 0.0, 0.15, 0.00, 0.01)

st.write("")
predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

# ----------------------------
# Predict
# ----------------------------
if predict_btn:
    payload = {
        "review_text": review_text,
        "rating": int(rating),
        "helpful_votes": int(helpful_votes),
        "verified_purchase": bool(verified_purchase),
        "verified_boost": float(verified_boost),
        "mode": mode,
        "text_weight": float(text_weight),
        "threshold": float(threshold),
    }

    if show_payload:
        st.subheader("Payload sent to API")
        st.json(payload)

    st.subheader("Result")

    with st.spinner("Calling API..."):
        out = None
        last_err = None
        status_code = None
        raw_text = None

        for attempt in range(1, 4):
            try:
                r = requests.post(predict_url, json=payload, timeout=30)
                status_code = r.status_code
                raw_text = r.text

                if r.status_code == 422:
                    st.error("422 validation error (payload rejected).")
                    st.code(raw_text)
                    break

                r.raise_for_status()
                out = r.json()
                break
            except Exception as e:
                last_err = e
                time.sleep(attempt)

        if out is None:
            st.error("Streamlit did NOT get a valid JSON response.")
            st.write("HTTP status:", status_code)
            if raw_text:
                st.code(raw_text)
            if last_err:
                st.write("Last error:", str(last_err))
            st.stop()

    # Show exactly what Streamlit received
    if show_response:
        st.subheader("Exact JSON Streamlit received")
        st.json(out)

    # Decide label using ONLY prediction from API
    pred_raw = out.get("prediction", None)
    try:
        pred = int(pred_raw)
    except Exception:
        pred = None

    ppos = out.get("probability_positive", None)

    st.caption(f"Parsed: prediction_raw={pred_raw} → prediction_int={pred} | probability_positive={ppos}")

    if pred == 1:
        st.success("✅ Positive (API prediction=1)")
    elif pred == 0:
        st.warning("⚠️ Negative (API prediction=0)")
    else:
        st.error("Could not parse prediction from API response.")
