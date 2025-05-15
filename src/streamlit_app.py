import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os
import PyPDF2
import matplotlib.pyplot as plt

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl")),
        "Ensemble": joblib.load(os.path.join(MODEL_DIR, "ensemble_model.pkl"), mmap_mode='r'),
        "SVM": joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl")),
    }


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def categorize_text(text):
    categories = {
        "Education": ["education", "degree", "university", "bachelor", "master", "phd"],
        "Experience": ["experience", "worked", "job", "position", "years"],
        "Requirement": ["requirement", "required", "criteria"],
        "Responsibility": ["responsibility", "tasks", "duty"],
        "Skill": ["skill", "expertise", "proficiency", "tools"],
        "SoftSkill": ["communication", "leadership", "teamwork", "problem-solving"],
    }
    counts = {}
    for category, keywords in categories.items():
        count = 0
        for kw in keywords:
            count += len(re.findall(rf"\b{kw}\b", text, flags=re.IGNORECASE))
        counts[category] = count
    return counts

st.title("CV Parsing and Text Classification")

models = load_models()

uploaded_file = st.file_uploader("Upload CV (PDF only):", type=["pdf"])

if uploaded_file is not None:
    cv_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted CV Text")
    st.text_area("", cv_text, height=200)

    st.subheader("Select Model")
    model_choice = st.selectbox("Choose model:", list(models.keys()))

    if st.button("Predict"):
        model = models[model_choice]
        pred = model.predict([cv_text])[0]
        st.success(f"Prediction: {pred}")

        st.subheader("Categorization Results")
        cat_counts = categorize_text(cv_text)
        df = pd.DataFrame(cat_counts.items(), columns=["Category", "Count"])
        st.dataframe(df)

        st.subheader("Category Distribution Pie Chart")
        fig, ax = plt.subplots()
        ax.pie(df["Count"], labels=df["Category"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)