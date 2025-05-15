import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os
import PyPDF2

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl")),
        "Ensemble" :joblib.load(os.path.join(MODEL_DIR, "ensemble_model.pkl")),
        "SVM": joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl")),
    }

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def categorize_text(text):
    # Define categories and keywords
    categories = {
        "Education": ["education", "degree", "university", "bachelor", "master", "phd"],
        "Experience": ["experience", "worked", "job", "position", "years"],
        "Requirement": ["requirement", "required", "criteria"],
        "Responsibility": ["responsibility", "tasks", "duty"],
        "Skill": ["skill", "expertise", "proficiency", "tools"],
        "SoftSkill": ["communication", "leadership", "teamwork", "problem-solving"],
    }

    counts = {cat: 0 for cat in categories}
    for cat, keywords in categories.items():
        for kw in keywords:
            counts[cat] += len(re.findall(rf"\b{kw}\b", text, flags=re.IGNORECASE))
    return counts

# Load models once
models = load_models()

st.title("CV Parsing and Text Classification App")

# Upload PDF
uploaded_file = st.file_uploader("Upload your CV (PDF only):", type=["pdf"])

cv_text = ""
if uploaded_file is not None:
    cv_text = extract_text_from_pdf(uploaded_file)
    if cv_text.strip():
        st.text_area("Extracted Text:", cv_text, height=200)
    else:
        st.error("Unable to extract text from the uploaded file.")

# Show category counts if text exists
if cv_text.strip():
    st.subheader("Categorization Results")
    counts = categorize_text(cv_text)

    # Format output like yang kamu mau
    output_lines = [
        f"Education       {counts['Education']}",
        f"    Experience   {counts['Experience']}",
        f"   Requirement {counts['Requirement']}",
        f"Responsibility    {counts['Responsibility']}",
        f"         Skill       {counts['Skill']}",
        f"     SoftSkill {counts['SoftSkill']}",
    ]
    formatted_output = "\n".join(output_lines)
    st.text_area("Category Counts:", formatted_output, height=150)

# Text input untuk prediksi (manual atau hasil extract)
st.subheader("Text Input for Classification")
text = st.text_area("Enter text or use extracted text:", value=cv_text if cv_text.strip() else "")

# Pilih model
model_choice = st.selectbox("Choose a model:", list(models.keys()))

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter or select some text.")
    else:
        model = models[model_choice]
        prediction = model.predict([text])[0]
        st.success(f"Prediction: {prediction}")