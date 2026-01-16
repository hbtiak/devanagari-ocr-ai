import streamlit as st
import numpy as np

from src.enhancement.enhance import enhance_image
from src.ocr.infer import run_ocr
from src.output.export import save_docx, save_pdf

st.set_page_config(page_title="Devanagari OCR AI", layout="wide")

st.title("📜 Devanagari Manuscript OCR")
st.caption("Reads handwritten or scanned Devanagari text and exports DOCX/PDF")

uploaded = st.file_uploader("Upload manuscript image", type=["png","jpg","jpeg"])

if uploaded:
    import cv2
    bytes_data = uploaded.read()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(img, channels="BGR")

    enhanced = enhance_image(img)
    st.subheader("Enhanced Image")
    st.image(enhanced, clamp=True)

    text, confidence = run_ocr(enhanced)
    st.subheader("Recognized Text")
    st.text_area("Output", text, height=200)

    st.subheader("Confidence")
    st.progress(min(confidence,1.0))
    st.write(f"{confidence:.2f}")

    if st.button("Export DOCX"):
        save_docx(text, "output.docx")
        st.success("Saved as output.docx")

    if st.button("Export PDF"):
        save_pdf(text, "output.pdf")
        st.success("Saved as output.pdf")
