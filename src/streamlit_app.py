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
    # Define categories and their respective keywords
    categories = {
        "Education": ["education", "degree", "university", "bachelor", "master", "phd"],
        "Experience": ["experience", "worked", "job", "position", "years"],
        "Requirement": ["requirement", "required", "criteria"],
        "Responsibility": ["responsibility", "tasks", "duty"],
        "Skill": ["skill", "expertise", "proficiency", "tools"],
        "SoftSkill": ["communication", "leadership", "teamwork", "problem-solving"],
    }

    # Count occurrences of keywords in the text
    category_counts = {category: 0 for category in categories.keys()}
    for category, keywords in categories.items():
        for keyword in keywords:
            category_counts[category] += len(re.findall(rf"\b{keyword}\b", text, flags=re.IGNORECASE))
    return category_counts

# Load models once
models = load_models()

st.title("CV Parsing and Text Classification App")

# Upload CV File
uploaded_file = st.file_uploader("Upload your CV (PDF format only):", type=["pdf"])

cv_text = ""  # initialize variable here so it is always defined

if uploaded_file is not None:
    cv_text = extract_text_from_pdf(uploaded_file)
    if cv_text.strip():
        st.text_area("Extracted Text:", cv_text, height=200)
    else:
        st.error("Unable to extract text from the uploaded file.")

# Categorization and Display
if cv_text.strip():
    st.subheader("Categorization Results")
    category_counts = categorize_text(cv_text)
    df = pd.DataFrame(list(category_counts.items()), columns=["Category", "Count"])
    st.table(df)

    # Pie Chart Visualization
    st.subheader("Category Distribution (Pie Chart)")
    fig, ax = plt.subplots()
    ax.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Text Input for Classification (either manual or from extracted text)
st.subheader("Text Input for Classification")
text = st.text_area("Enter text manually or use the extracted text above:", value=cv_text if cv_text.strip() else "")

# Model Selection and Prediction
model_choice = st.selectbox("Choose a model:", list(models.keys()))

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter or select some text.")
    else:
        model = models[model_choice]
        prediction = model.predict([text])[0]
        st.success(f"Prediction: {prediction}")