import streamlit as st
import os

def show_model_performance():

    st.header("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "94.2%")
    col2.metric("Precision", "93.8%")
    col3.metric("Recall", "94.1%")
    col4.metric("F1 Score", "93.9%")

    st.divider()

    if os.path.exists("assets/training_accuracy.png"):
        st.subheader("Training Accuracy")
        st.image(
            "assets/training_accuracy.png",
            use_container_width=True
        )

    if os.path.exists("assets/training_loss.png"):
        st.subheader("Training Loss")
        st.image(
            "assets/training_loss.png",
            use_container_width=True
        )