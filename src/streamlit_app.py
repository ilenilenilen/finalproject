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
    # Label utama saja
    categories = {
        "Education": ["education", "degree", "university", "bachelor", "master", "phd", "gpa"],
        "Experience": ["experience", "worked", "job", "position", "years", "intern"],
        "Requirement": ["requirement", "mandatory", "qualification", "criteria", "must", "eligibility"],
        "Responsibility": ["responsibility", "task", "duty", "role", "accountable", "responsible"],
        "Skill": [
            "skill", "expertise", "proficiency", "tools", "excel",
            "project management", "research", "problem solving", "public speaking",
            "machine learning", "data analysis", "feature engineering", "model deployment",
            "algorithm", "python", "r", "sql", "tensorflow", "pytorch"
        ],
        "SoftSkill": ["communication", "leadership", "teamwork", "problem-solving", "advocacy", "relationship building"],
    }

    raw_sentences = tokenizer.tokenize(text)
    sentences = []
    for s in raw_sentences:
        sentences.extend(re.split(r"[\n\u2022\-]+", s))
    sentences = [s.strip() for s in sentences if s.strip()]

    categorized_sentences = []
    combined_text = ""
    current_category = None

    for sent in sentences:
        sent_lower = sent.lower()
        matched_category = None

        for category, keywords in categories.items():
            if any(re.search(rf"\b{kw}\b", sent_lower) for kw in keywords):
                matched_category = category
                break

        if matched_category:
            if current_category and current_category == matched_category:
                combined_text += " " + sent
            else:
                if combined_text:
                    categorized_sentences.append({"text": combined_text, "category": current_category})
                combined_text = sent
                current_category = matched_category
        else:
            if current_category:
                combined_text += " " + sent

    if combined_text:
        categorized_sentences.append({"text": combined_text, "category": current_category})

    return categorized_sentences


def check_senior_data_scientist_criteria(text):
    """
    Fungsi ini mengecek apakah CV memenuhi kriteria minimal seorang Senior Data Scientist
    berdasarkan beberapa kata kunci penting.
    """
    keywords_mandatory = [
        "master", "phd", "machine learning", "model deployment",
        "feature engineering", "data analysis", "python", "statistics",
        "5 years experience", "senior", "leadership", "project management"
    ]

    text_lower = text.lower()
    missing_keywords = [kw for kw in keywords_mandatory if kw not in text_lower]

    if len(missing_keywords) == 0:
        return True, "CV memenuhi kriteria Senior Data Scientist."
    else:
        return False, f"CV kurang beberapa kriteria penting: {', '.join(missing_keywords)}"


def match_cv_with_jd(cv_categories, jd_text):
    jd_categories = categorize_sentences(jd_text)

    # Create a DataFrame to compare
    cv_df = pd.DataFrame(cv_categories)
    jd_df = pd.DataFrame(jd_categories)

    # Check matches
    match_results = []
    for _, jd_row in jd_df.iterrows():
        jd_text = jd_row["text"].lower()
        jd_category = jd_row["category"]

        match_count = cv_df[(cv_df["category"] == jd_category) & (cv_df["text"].str.lower().str.contains(jd_text))].shape[0]

        recommendation = (
            "Good match" if match_count > 0 else "Consider adding more details for this category in the CV"
        )

        match_results.append({
            "JD Category": jd_category,
            "JD Text": jd_row["text"],
            "Matches in CV": match_count,
            "Recommendation": recommendation
        })

    return pd.DataFrame(match_results)


# --- STREAMLIT APP ---

st.title("📄 CV Parsing and Job Description Matching")
st.markdown("Easily extract and categorize text from CVs, and match them against job descriptions (JDs).")

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

st.subheader("Job Description Input")
jd_text = st.text_area("Enter the Job Description:")

if st.button("Analyze and Match"):
    if not cv_text.strip():
        st.warning("Please upload a CV.")
    elif not jd_text.strip():
        st.warning("Please enter a Job Description.")
    else:
        # Categorize CV text
        cv_categories = categorize_sentences(cv_text)

        # Match CV with JD
        match_df = match_cv_with_jd(cv_categories, jd_text)

        st.subheader("Matching Results")
        st.dataframe(match_df.style.highlight_max(subset=["Matches in CV"], color="lightgreen", axis=0))

        # Check senior data scientist criteria
        is_senior, message = check_senior_data_scientist_criteria(cv_text)
        if is_senior:
            st.success(message)
        else:
            st.warning(message)

        # Optional visualization: Matching Distribution
        st.subheader("Matching Distribution")
        fig, ax = plt.subplots()
        match_df["Matches in CV"].plot(kind="bar", ax=ax, color="skyblue", alpha=0.7, edgecolor="black")
        ax.set_title("Number of Matches per JD Category")
        ax.set_xlabel("Job Description Categories")
        ax.set_ylabel("Matches in CV")
        ax.set_xticks(range(len(match_df)))
        ax.set_xticklabels(match_df["JD Category"], rotation=45, ha="right")
        st.pyplot(fig)
