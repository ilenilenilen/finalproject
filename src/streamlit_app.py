import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os


# Path yang benar: gunakan direktori saat ini (tempat streamlit_app.py berada)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl")),
        "Ensemble" :joblib.load(os.path.join(MODEL_DIR, "ensemble_model.pkl")),
        "SVM": joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl")),
    }


models = load_models()

st.title("Text Classification App")
text = st.text_area("Enter some text:")
model_choice = st.selectbox("Choose a model:", list(models.keys()))

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        model = models[model_choice]
        prediction = model.predict([text])[0]
        st.success(f"Prediction: {prediction}")
