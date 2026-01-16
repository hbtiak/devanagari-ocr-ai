import streamlit as st
import numpy as np

from src.enhancement.enhance import enhance_image
from src.ocr.infer import run_ocr
from src.language_model.correct import correct_text

st.set_page_config(page_title="Devanagari OCR AI", layout="wide")

st.title("📜 Devanagari Manuscript OCR")
st.caption("AI system for reading damaged handwritten Devanagari text")

uploaded = st.file_uploader("Upload manuscript image", type=["png","jpg","jpeg"])

if uploaded:
    import cv2  # lazy import – REQUIRED for Streamlit Cloud

    bytes_data = uploaded.read()
    img = cv2.imdecode(
        np.frombuffer(bytes_data, np.uint8),
        cv2.IMREAD_COLOR
    )

    enhanced = enhance_image(img)
    text, confidence = run_ocr(enhanced)
    final_text = correct_text(text)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(img, channels="BGR")
        st.subheader("Enhanced")
        st.image(enhanced, clamp=True)

    with col2:
        st.subheader("Recognized Text")
        st.text_area("Output", final_text, height=200)
        st.subheader("Confidence")
        st.progress(min(confidence, 1.0))
        st.write(f"{confidence:.2f}")

        if confidence < 0.75:
            st.warning("Low confidence — human review recommended")
        else:
            st.success("High confidence output")
