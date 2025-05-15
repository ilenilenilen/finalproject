import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os
import PyPDF2
import tempfile
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


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def categorize_text(text):
    """Categorize text into predefined categories based on keywords."""
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

# Streamlit App
st.title("CV Parsing and Text Classification App")

# Load models
models = load_models()

# File uploader
uploaded_file = st.file_uploader("Upload your CV (PDF format only):", type=["pdf"])
cv_text = ""

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            cv_text = extract_text_from_pdf(temp_file_path)
        
        if cv_text.strip():
            st.text_area("Extracted Text:", cv_text, height=200)
        else:
            st.error("No text could be extracted from the uploaded file.")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Model selection and prediction
st.subheader("Text Input for Classification")
text = st.text_area("Enter text manually or use the extracted text above:", value=cv_text if cv_text.strip() else "")
model_choice = st.selectbox("Choose a model:", list(models.keys()))

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter or select some text.")
    else:
        # Perform prediction
        model = models[model_choice]
        prediction = model.predict([text])[0]
        st.success(f"Prediction: {prediction}")
        
        # Categorization
        st.subheader("Categorization Results")
        category_counts = categorize_text(text)
        df = pd.DataFrame(list(category_counts.items()), columns=["Category", "Count"])
        
        # Display table and chart together
        col1, col2 = st.columns(2)
        with col1:
            st.table(df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)