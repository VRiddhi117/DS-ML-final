# streamlit/app.py
import os
import time
import requests
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Cosmetics Review Classifier", layout="wide")

# ----------------------------
# Defaults
# ----------------------------
DEFAULT_API_URL = os.getenv("API_URL", "https://ds-ml-final.onrender.com").rstrip("/")
BG_URL = os.getenv(
    "BG_URL",
    "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?auto=format&fit=crop&w=1920&q=80",
)

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    f"""
<style>
/* Background */
[data-testid="stAppViewContainer"] {{
  background: url("{BG_URL}") center/cover fixed no-repeat;
  position: relative;
}}
[data-testid="stAppViewContainer"]::before {{
  content: "";
  position: absolute;
  inset: 0;
  background: rgba(8,10,14,0.88);
  backdrop-filter: blur(7px);
  -webkit-backdrop-filter: blur(7px);
  z-index: 0;
}}
[data-testid="stAppViewContainer"] > .main {{
  position: relative;
  z-index: 1;
}}

/* Layout */
.block-container {{
  max-width: 1150px;
  padding-top: 1.1rem;
  padding-bottom: 2rem;
}}

/* Glass card */
.glass {{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px 20px;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  box-shadow: 0 14px 38px rgba(0,0,0,0.38);
}}
.hero-title {{
  font-size: 42px;
  font-weight: 900;
  margin: 0;
  letter-spacing: 0.2px;
}}
.hero-sub {{
  font-size: 15px;
  opacity: 0.9;
  margin-top: 10px;
  line-height: 1.35;
}}
.badges {{
  display:flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}}
.badge {{
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.12);
  font-size: 12px;
  opacity: 0.95;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background: rgba(14,16,22,0.78);
  border-right: 1px solid rgba(255,255,255,0.08);
}}

/* Inputs */
div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input {{
  background: rgba(20,24,32,0.92) !important;
  border-radius: 12px !important;
  color: #e8eefc !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}}
div[data-testid="stSlider"] > div {{
  padding-top: 0.25rem;
}}

/* Buttons */
button[kind="primary"] {{
  border-radius: 14px !important;
  padding: 0.8rem 1rem !important;
}}
button {{
  border-radius: 14px !important;
}}

/* Result cards */
.result {{
  border-radius: 18px;
  padding: 16px 16px;
  border: 1px solid rgba(255,255,255,0.10);
}}
.result-pos {{
  background: rgba(34,197,94,0.12);
  border: 1px solid rgba(34,197,94,0.22);
}}
.result-neg {{
  background: rgba(239,68,68,0.12);
  border: 1px solid rgba(239,68,68,0.22);
}}
.small-muted {{
  opacity: 0.82;
  font-size: 13px;
}}
.kpi {{
  display:flex;
  gap: 14px;
  flex-wrap: wrap;
  margin-top: 10px;
}}
.kpi > div {{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 10px 12px;
  border-radius: 14px;
  min-width: 150px;
}}
.kpi-label {{
  font-size: 12px;
  opacity: 0.75;
}}
.kpi-value {{
  font-size: 18px;
  font-weight: 800;
  margin-top: 2px;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
<div class="glass">
  <div class="badges">
    <div class="badge">Text • Sentiment</div>
    <div class="badge">Rating • Score</div>
    <div class="badge">Hybrid • Blend</div>
    <div class="badge">FastAPI + Render</div>
  </div>
  <p class="hero-title">Cosmetics Review Classifier</p>
  <p class="hero-sub">
    Predict whether a review is <b>Positive</b> or <b>Negative</b> using review text + rating signals.
    Includes API health check, explanation, and raw JSON view for demos.
  </p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# ----------------------------
# Sidebar: API settings
# ----------------------------
with st.sidebar:
    st.subheader("API Settings")

    api_url = st.text_input(
        "API base URL",
        value=st.session_state.get("api_url", DEFAULT_API_URL),
        help="Local: http://127.0.0.1:8000 | Render: https://xxxx.onrender.com",
    ).strip().rstrip("/")

    st.session_state["api_url"] = api_url
    predict_url = f"{api_url}/predict"

    colA, colB = st.columns(2)
    with colA:
        health_btn = st.button("Health check", use_container_width=True)
    with colB:
        wake_btn = st.button("Wake", use_container_width=True)

    if health_btn or wake_btn:
        try:
            r = requests.get(api_url, timeout=15)
            st.success(f"GET / → {r.status_code}")
            try:
                st.json(r.json())
            except Exception:
                st.code(r.text)
        except Exception as e:
            st.error(f"GET / failed: {e}")

    st.caption("Predict endpoint:")
    st.code(predict_url)

    st.divider()
    st.subheader("Display")
    show_explain = st.checkbox("Show explanation", value=True)
    show_raw = st.checkbox("Show raw API JSON", value=False)
    show_payload = st.checkbox("Show payload", value=False)

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Review inputs")
    review_text = st.text_area("Review text", value="Amazing moisturizer", height=160)
    rating = st.slider("Rating (1–5)", 1, 5, 2)
    helpful_votes = st.number_input("Helpful votes", min_value=0, value=0, step=1)
    verified_purchase = st.checkbox("Verified purchase", value=True)

with right:
    st.subheader("Mode")
    mode = st.radio(
        "Prediction mode",
        ["text", "rating", "hybrid"],
        index=0,
        format_func=lambda x: {
            "text": "Text-only (sentiment)",
            "rating": "Rating-only (score)",
            "hybrid": "Hybrid (blend)",
        }[x],
    )

    if mode == "hybrid":
        text_weight = st.slider("Text weight (hybrid)", 0.0, 1.0, 0.80, 0.05)
        st.caption("Higher = trust text more.")
    elif mode == "text":
        text_weight = 1.0
        st.caption("Text-only ignores rating.")
    else:
        text_weight = 0.0
        st.caption("Rating-only ignores text.")

    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.05)
    verified_boost = st.slider("Verified purchase boost", 0.0, 0.15, 0.00, 0.01)

st.write("")
predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

# ----------------------------
# Predict
# ----------------------------
def safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

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
        st.subheader("Payload")
        st.json(payload)

    with st.spinner("Calling API..."):
        out = None
        last_err = None
        status_code = None
        raw_text = None

        for attempt in range(1, 4):
            try:
                r = requests.post(predict_url, json=payload, timeout=45)
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
                time.sleep(1.5 * attempt)

        if out is None:
            st.error("Prediction failed.")
            st.caption(f"POST {predict_url}")
            if status_code is not None:
                st.caption(f"HTTP status: {status_code}")
            if raw_text:
                st.code(raw_text)
            if last_err:
                st.caption(f"Last error: {last_err}")
            st.stop()

    # Parse
    pred = safe_int(out.get("prediction", None))
    p_pos = safe_float(out.get("probability_positive", out.get("probability", float("nan"))))
    exp = out.get("explain", {}) if isinstance(out.get("explain", {}), dict) else {}

    if pred is None:
        pred = 1 if (p_pos == p_pos and p_pos >= threshold) else 0

    is_pos = (pred == 1)
    label = "Positive" if is_pos else "Negative"
    emoji = "✅" if is_pos else "⚠️"
    conf = (p_pos if is_pos else (1 - p_pos)) if (p_pos == p_pos) else 0.5

    st.subheader("Result")

    st.markdown(
        f"""
        <div class="result {'result-pos' if is_pos else 'result-neg'}">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
            <div style="font-size:20px;"><b>{emoji} {label}</b></div>
            <div class="small-muted"><b>Confidence</b>: {conf*100:.1f}%</div>
          </div>
          <div class="small-muted" style="margin-top:6px;">
            Decision rule: p_positive ({p_pos:.3f}) {'≥' if p_pos >= threshold else '<'} threshold ({threshold:.2f})
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(max(0.0, min(1.0, conf)))

    # KPI row
    st.markdown(
        f"""
        <div class="kpi">
          <div><div class="kpi-label">Mode</div><div class="kpi-value">{out.get('mode', mode)}</div></div>
          <div><div class="kpi-label">p_positive</div><div class="kpi-value">{p_pos:.3f}</div></div>
          <div><div class="kpi-label">Threshold</div><div class="kpi-value">{threshold:.2f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_explain:
        st.write("")
        st.subheader("Explanation")
        if exp:
            st.write(
                f"- p_text: **{exp.get('p_text', 'N/A')}**\n"
                f"- p_rating: **{exp.get('p_rating', 'N/A')}**\n"
                f"- p_final: **{exp.get('p_final', p_pos)}**\n"
                f"- verified_purchase: **{exp.get('verified_purchase', verified_purchase)}**\n"
                f"- boost_applied: **{exp.get('boost_applied', 0.0)}**"
            )
            if exp.get("negation_detected") is True:
                st.warning("Negation detected in text (e.g., 'not good', 'not worth'). Text score adjusted.")
            if exp.get("model_loaded") is False:
                st.warning("Text model not loaded on API — using fallback heuristic.")
        else:
            st.caption("No explanation returned by API.")

    if show_raw:
        st.write("")
        with st.expander("Raw API response"):
            st.json(out)
