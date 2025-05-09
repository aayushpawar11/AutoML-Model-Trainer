import os
import streamlit as st
import requests
import pandas as pd

st.title("🧠 AutoML Model Trainer")

st.markdown("Upload a CSV file and discover the best ML model using Optuna optimization.")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    with st.spinner("Training... please wait ⏳"):
        # Send the file to your Flask API
        API_URL = os.getenv("API_URL", "http://localhost:5050/train")
        response = requests.post(API_URL, files={"file": uploaded_file})

    if response.ok:
        result = response.json()
        st.success("Training complete!")

        st.metric(label="Model", value=result["model"])
        st.metric(label="Accuracy", value=round(result["accuracy"] * 100, 2))
        st.subheader("Best Hyperparameters")
        st.json(result["hyperparameters"])
    else:
        st.error(f"Failed to train model: {response.json().get('error', 'Unknown error')}")
