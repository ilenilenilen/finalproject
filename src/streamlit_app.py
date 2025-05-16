import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os
import PyPDF2
import matplotlib.pyplot as plt

# Set writable directories for Streamlit and Matplotlib
os.environ["STREAMLIT_CONFIG_DIR"] = "/tmp/.streamlit"
os.environ["MPLCONFIGDIR"] = "/tmp/.matplotlib"

#MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    model_dir = os.path.join(os.getcwd(), "models")
    return {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl")),
        "Ensemble": joblib.load(os.path.join(MODEL_DIR, "ensemble_model.pkl"), mmap_mode='r'),
        "SVM": joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl")),
    }


def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def categorize_text(text):
    categories = {
        "Education": ["education", "degree", "university", "bachelor", "master", "phd"],
        "Experience": ["experience", "worked", "job", "position", "years"],
        "Requirement": ["requirement", "required", "criteria"],
        "Responsibility": ["responsibility", "tasks", "duty"],
        "Skill": ["skill", "expertise", "proficiency", "tools"],
        "SoftSkill": ["communication", "leadership", "teamwork", "problem-solving"],
    }
    category_counts = {category: 0 for category in categories.keys()}
    for category, keywords in categories.items():
        for keyword in keywords:
            category_counts[category] += len(re.findall(rf"\b{keyword}\b", text, flags=re.IGNORECASE))
    return category_counts

st.title("CV Parsing and Text Classification App")

models = load_models()

uploaded_file = st.file_uploader("Upload your CV (PDF format only):", type=["pdf"])
cv_text = ""

if uploaded_file is not None:
    try:
        with uploaded_file:
            cv_text = extract_text_from_pdf(uploaded_file)
        if cv_text.strip():
            st.text_area("Extracted Text:", cv_text, height=200)
        else:
            st.error("No text could be extracted from the uploaded file.")
    except Exception as e:
        st.error(f"Error processing file: {e}")

if cv_text.strip():
    st.subheader("Category Distribution (Pie Chart)")
    category_counts = categorize_text(cv_text)
    total_count = sum(category_counts.values())
    labels = [
        f"{category} ({count} - {count / total_count * 100:.1f}%)"
        for category, count in category_counts.items()
    ]
    fig, ax = plt.subplots()
    ax.pie(category_counts.values(), labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

st.subheader("Text Input for Classification")
text = st.text_area("Enter text manually or use the extracted text above:", value=cv_text if cv_text.strip() else "")

model_choice = st.selectbox("Choose a model:", list(models.keys()))

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter or select some text.")
    else:
        model = models[model_choice]
        prediction = model.predict([text])[0]
        st.success(f"Prediction: {prediction}")