import streamlit as st
import numpy as np
import os
from PIL import Image

from config import CLASS_NAMES
from prediction import predict
from gradcam import generate_gradcam
from report_generator import generate_report
from model_performance import show_model_performance

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="🫁 LungVision AI",
    page_icon="🫁",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.main{
    padding-top:2rem;
}

h1{
    color:#1565C0;
}

.stButton>button{
    width:100%;
    height:50px;
    font-size:18px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🫁 LungVision AI")

st.sidebar.markdown("""
### About

LungVision AI is a Deep Learning application that classifies Chest X-Ray images into:

- 🦠 COVID
- 🫁 Lung Opacity
- ✅ Normal
- 🦠 Viral Pneumonia

---

### Model
Custom CNN

---

### Input Size
224 × 224

---

### Framework
TensorFlow + Streamlit

---

### Explainability
Grad-CAM Heatmaps
""")
st.sidebar.markdown("---")

st.sidebar.success("✅ Model Loaded")

st.sidebar.info("""
**Developed By**

Wariz Shaik

Final Year Mini Project
""")

# ======================================================
# HEADER
# ======================================================

st.title("🫁 LungVision AI")

st.markdown("""
### AI Powered Chest X-Ray Disease Detection

Upload a Chest X-ray image to classify lung disease using a Convolutional Neural Network (CNN).

After prediction, you can generate a **Grad-CAM Heatmap** to visualize the regions used by the AI model.
""")

st.divider()

# ======================================================
# FILE UPLOADER
# ======================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

# ======================================================
# PREDICTION
# ======================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image.save("temp_xray.jpg")

    prediction, predicted_index, confidence = predict(image)

    col1, col2 = st.columns([1,1])

    # ---------------- LEFT ----------------

    with col1:

        st.image(
            image,
            caption="Uploaded Chest X-ray",
            use_container_width=True
        )

    # ---------------- RIGHT ----------------

    with col2:

        st.success(f"### Prediction\n\n{CLASS_NAMES[predicted_index]}")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(confidence/100)

        st.markdown("### Class Probabilities")

        probabilities = {}

        for i, cls in enumerate(CLASS_NAMES):

            prob = prediction[0][i] * 100

            probabilities[cls] = prob

            st.write(cls)

            st.progress(float(prediction[0][i]))

            st.write(f"{prob:.2f}%")

    st.divider()

    # ======================================================
    # GRAD-CAM
    # ======================================================

    st.subheader("🔥 Grad-CAM Explainability")

    if st.button("Generate Grad-CAM Heatmap"):

        with st.spinner("Generating Heatmap..."):

            generate_gradcam()

        if os.path.exists("src/gradcam_result.png"):

            st.success("Grad-CAM Generated Successfully!")

            c1, c2 = st.columns(2)

            with c1:

                st.image(
                    image,
                    caption="Original X-ray",
                    use_container_width=True
                )

            with c2:

                st.image(
                    "src/gradcam_result.png",
                    caption="Grad-CAM Heatmap",
                    use_container_width=True
                )

            st.info("""
Red and Yellow regions indicate the regions that influenced the CNN's prediction the most.

Blue regions contributed very little.
""")

        else:

            st.error("Heatmap could not be generated.")

    st.divider()

    # ======================================================
    # PDF REPORT
    # ======================================================

    st.subheader("📄 Report")

    if st.button("Generate PDF Report"):

        generate_report(
            CLASS_NAMES[predicted_index],
            confidence,
            probabilities
        )

        with open("Prediction_Report.pdf", "rb") as pdf:

            st.download_button(
                label="⬇ Download Prediction Report",
                data=pdf,
                file_name="Prediction_Report.pdf",
                mime="application/pdf"
            )

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.markdown("""
<center>

### 🫁 LungVision AI

AI Powered Chest X-Ray Disease Detection using Deep Learning

**Final Year Mini Project**

TensorFlow • Keras • Streamlit • Grad-CAM

</center>
""", unsafe_allow_html=True)
st.divider()

if st.checkbox("📊 Show Model Performance"):

    show_model_performance()
st.warning(
    "⚠ This application is intended for educational purposes only and should not replace professional medical diagnosis."
)
with st.expander("ℹ About This Project"):

    st.write("""
**LungVision AI** is a Deep Learning application developed to classify Chest X-ray images into four categories:

• COVID-19

• Lung Opacity

• Normal

• Viral Pneumonia

The application uses a custom Convolutional Neural Network (CNN) and Grad-CAM Explainability to highlight image regions responsible for predictions.

This project was developed for educational purposes as a Final Year Mini Project.
""")