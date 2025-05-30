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
import io

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

# Categorize sentences based on predefined categories
def categorize_sentences(text):
    categories = {
        "Education": [
            "education", "degree", "university", "bachelor", "master", "phd", "gpa",
            "computer science", "mathematics", "statistics", "information system", "relevant major"
        ],
        "Experience": [
            "experience", "worked", "job", "position", "years", "intern",
            "5 years", "data science", "data scientist", "fintech", "finance services",
            "production environment"
        ],
        "Requirement": [
            "requirement", "mandatory", "qualification", "criteria", "must", "eligibility",
            "deep understanding", "strong analytical thinking", "proven experience", "advantage"
        ],
        "Responsibility": [
            "responsibility", "task", "duty", "role", "accountable", "responsible",
            "design", "build", "deploy", "perform testing", "model implementation",
            "fine tuning", "drive improvement"
        ],
        "Skill": [
            "skill", "expertise", "proficiency", "tools", "excel",
            "project management", "research", "problem solving", "public speaking",
            "machine learning", "model development", "model deployment", "risk evaluation",
            "business impact analysis", "feature engineering", "algorithm", "analysis"
        ],
        "SoftSkill": [
            "communication", "leadership", "teamwork", "problem-solving", "advocacy",
            "relationship building", "analytical thinking"
        ],
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
                categorized_sentences.append({"text": sent, "category": cat, "match_with_job_desc": False})

    return categorized_sentences

# Convert DataFrame to Excel
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Categorized Data")
    processed_data = output.getvalue()
    return processed_data

# STREAMLIT APP
st.title("CV Parsing with Manual Matching")

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

st.subheader("Text Input for Classification")
text_for_classification = st.text_area(
    "Enter CV text manually or use extracted CV text above:",
    value=cv_text if cv_text.strip() else "",
    height=200,
)

if st.button("Categorize and Match"):
    if not text_for_classification.strip():
        st.warning("Please enter or select some CV text for classification.")
    else:
        categorized = categorize_sentences(text_for_classification)
        df_categorized = pd.DataFrame(categorized)

        if df_categorized.empty:
            st.info("No categorized sentences found in the CV text.")
        else:
            st.subheader("Categorized CV Sentences with Manual Matching")
            for i in range(len(df_categorized)):
                checkbox = st.checkbox(
                    f"{df_categorized.iloc[i]['text']} ({df_categorized.iloc[i]['category']})",
                    key=f"match_{i}"
                )
                df_categorized.at[i, "match_with_job_desc"] = checkbox

            st.download_button(
                label="Download Categorized Data as Excel",
                data=convert_df_to_excel(df_categorized),
                file_name="categorized_cv_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Summary
            st.subheader("Summary of CV Categories")
            category_counts = df_categorized["category"].value_counts()
            for i, (cat, count) in enumerate(category_counts.items(), start=1):
                st.markdown(f"{i}. **{cat}**: {count}")

            # Pie Chart
            st.subheader("Category Distribution (Pie Chart)")
            fig, ax = plt.subplots()
            ax.pie(
                category_counts.values,
                labels=category_counts.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.axis("equal")
            st.pyplot(fig)
