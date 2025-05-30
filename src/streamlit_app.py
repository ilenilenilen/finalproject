# Import libraries
import re
import os
import joblib
import PyPDF2
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

# Download 'punkt' tokenizer
nltk.download('punkt', quiet=True)

# Initialize the Punkt tokenizer explicitly
tokenizer = PunktSentenceTokenizer()

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl")),
        "Ensemble": joblib.load(os.path.join(MODEL_DIR, "ensemble_model.pkl"), mmap_mode='r'),
        "SVM": joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl")),
    }

# Extract file from PDF
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
        "Education": ["education", "degree", "university", "bachelor", "master", "phd", "gpa", "computer science", "mathematics"],
        "Experience": ["experience", "worked", "job", "position", "years", "intern", "data science", "data scientist", "production environment"],
        "Requirement": ["requirement", "mandatory", "qualification", "criteria", "eligibility", "proven experience", "advantage"],
        "Responsibility": ["responsibility", "task", "duty", "role", "design", "build", "deploy", "testing", "model implementation"],
        "Skill": ["skill", "expertise", "tools", "excel", "problem solving", "machine learning", "model deployment", "algorithm"],
        "SoftSkill": ["communication", "leadership", "teamwork", "problem-solving", "analytical thinking"]
    }

    sentences = tokenizer.tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    categorized_sentences = []
    for sent in sentences:
        sent_lower = sent.lower()
        matched_categories = []
        for category, keywords in categories.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", sent_lower) for kw in keywords):
                matched_categories.append(category)
        if matched_categories:
            for cat in matched_categories:
                categorized_sentences.append({"text": sent, "category": cat})

    return categorized_sentences

# STREAMLIT APP
st.title("📄 CV Parsing with Manual Match Selection")

models = load_models()

uploaded_file = st.file_uploader("Upload your CV (PDF format only):", type=["pdf"])

cv_text = ""
if uploaded_file is not None:
    with uploaded_file:
        cv_text = extract_text_from_pdf(uploaded_file)
    if cv_text.strip():
        st.text_area("Extracted CV Text:", cv_text, height=200)
    else:
        st.error("No text could be extracted from the uploaded file.")

if st.button("Categorize and Analyze"):
    if not cv_text.strip():
        st.warning("Please upload a CV to analyze.")
    else:
        categorized = categorize_sentences(cv_text)
        df_categorized = pd.DataFrame(categorized)

        if df_categorized.empty:
            st.info("No categorized sentences found in the uploaded CV.")
        else:
            df_categorized["match"] = False  # Default value for the match column

            st.subheader("CV Categorization with Manual Match Selection")
            edited_df = st.experimental_data_editor(df_categorized, use_container_width=True)

            st.download_button(
                "Download Categorized CV as Excel",
                data=edited_df.to_csv(index=False),
                file_name="categorized_cv.xlsx",
                mime="text/csv",
            )

            # Summary and Pie Chart for CV Text
            st.subheader("Summary of CV Categories")
            df_cat = edited_df["category"].value_counts()
            st.bar_chart(df_cat)

            st.subheader("Category Distribution (Pie Chart)")
            fig, ax = plt.subplots()
            colors = list(mcolors.TABLEAU_COLORS.values())[:len(df_cat)]
            ax.pie(df_cat.values, labels=df_cat.index, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
