import re
import os
import joblib
import PyPDF2
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk.data

# Patch nltk.data.find to replace 'punkt_tab' with 'punkt'
_original_find = nltk.data.find

def patched_find(resource_name, *args, **kwargs):
    if 'punkt_tab' in resource_name:
        resource_name = resource_name.replace('punkt_tab', 'punkt')
    return _original_find(resource_name, *args, **kwargs)

nltk.data.find = patched_find

import nltk
from nltk.tokenize import sent_tokenize

# Download 'punkt' resource if not available yet
nltk.download('punkt', quiet=True)

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

def categorize_sentences(text):
    categories = {
        "Education": ["education", "degree", "university", "bachelor", "master", "phd"],
        "Experience": ["experience", "worked", "job", "position", "years"],
        "Requirement": ["requirement", "required", "criteria"],
        "Responsibility": ["responsibility", "tasks", "duty"],
        "Skill": ["skill", "expertise", "proficiency", "tools"],
        "SoftSkill": ["communication", "leadership", "teamwork", "problem-solving"],
    }
    
    sentences = sent_tokenize(text)
    categorized_sentences = []

    for sent in sentences:
        sent_lower = sent.lower()
        sent_categories = []
        for category, keywords in categories.items():
            if any(re.search(rf"\b{kw}\b", sent_lower) for kw in keywords):
                sent_categories.append(category)
        if not sent_categories:
            sent_categories = ["Uncategorized"]
        categorized_sentences.append({"sentence": sent, "categories": sent_categories})

    return categorized_sentences

# --- STREAMLIT APP ---

st.title("CV Parsing and Text Classification")

models = load_models()

uploaded_file = st.file_uploader("Upload your CV (PDF format only):", type=["pdf"])

cv_text = ""

if uploaded_file is not None:
    with uploaded_file:
        cv_text = extract_text_from_pdf(uploaded_file)
    if cv_text.strip():
        st.text_area("Extracted Text:", cv_text, height=200)
    else:
        st.error("No text could be extracted from the uploaded file.")

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

        st.subheader("Categorization Per Sentence")
        categorized = categorize_sentences(text)
        for item in categorized:
            st.markdown(f"- **Kalimat:** {item['sentence']}")
            st.markdown(f"  - **Kategori:** {', '.join(item['categories'])}")

        st.subheader("Category Distribution (Pie Chart)")
        all_categories = [cat for item in categorized for cat in item['categories']]
        df_cat = pd.Series(all_categories).value_counts()

        fig, ax = plt.subplots()
        ax.pie(df_cat.values, labels=df_cat.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
