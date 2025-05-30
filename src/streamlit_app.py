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
        "Education": ["education", "degree", "university", "bachelor", "master", "phd", "gpa"],
        "Experience": ["experience", "worked", "job", "position", "years", "intern"],
        "Requirement": ["requirement", "qualification", "mandatory", "eligibility"],
        "Responsibility": ["responsibility", "task", "duty", "role", "accountable"],
        "Skill": ["skill", "expertise", "tools", "excel", "machine learning"],
        "SoftSkill": ["communication", "leadership", "teamwork", "problem-solving"]
    }

    sentences = tokenizer.tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    categorized_sentences = []

    for sent in sentences:
        sent_lower = sent.lower()
        matched_categories = []
        for category, keywords in categories.items():
            if any(re.search(rf"\\b{re.escape(kw)}\\b", sent_lower) for kw in keywords):
                matched_categories.append(category)
        if matched_categories:
            for cat in matched_categories:
                categorized_sentences.append({"text": sent, "category": cat, "match": False})

    return categorized_sentences

# STREAMLIT APP
st.title("\ud83d\udcc4 CV Parsing with Job Description Matching")

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

st.subheader("Job Description Input (from HR)")
job_desc = st.text_area("Enter Job Description (responsibilities, tasks, etc.):", height=150)
job_qual = st.text_area("Enter Job Qualifications:", height=150)

st.subheader("Text Input for Classification")
text_for_classification = st.text_area(
    "Enter CV text manually or use extracted CV text above:",
    value=cv_text if cv_text.strip() else "",
    height=200,
)

model_choice = st.selectbox("Choose a model:", list(models.keys()))

if st.button("Predict and Match"):
    if not text_for_classification.strip():
        st.warning("Please enter or select some CV text for classification.")
    else:
        model = models[model_choice]
        prediction = model.predict([text_for_classification])[0]
        st.success(f"Prediction: {prediction}")

        categorized = categorize_sentences(text_for_classification)

        # Convert to DataFrame for display
        df_categorized = pd.DataFrame(categorized)
        df_categorized['Job Description'] = job_desc
        df_categorized['Job Qualifications'] = job_qual

        if df_categorized.empty:
            st.info("No categorized sentences from CV.")
        else:
            st.dataframe(
                df_categorized,
                use_container_width=True
            )

            # Allow editing the 'match' column
            edited_df = st.experimental_data_editor(df_categorized, num_rows="dynamic")

            # Add download button
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(edited_df)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="cv_analysis.csv",
                mime="text/csv"
            )

            # Summary of CV Categories
            st.subheader("Summary of CV Categories")
            summary = edited_df['category'].value_counts()

            for i, (cat, count) in enumerate(summary.items(), start=1):
                color = mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS.keys())[i % len(mcolors.TABLEAU_COLORS)]]
                st.markdown(f"""
                    <div style="background-color: {color}; border-radius: 10px; padding: 10px; margin-bottom: 10px; color: white;">
                        <strong>{i}. {cat}</strong>: {count}
                    </div>
                """, unsafe_allow_html=True)

            # Pie Chart for CV Categories
            st.subheader("Category Distribution (Pie Chart) for CV Text")
            fig, ax = plt.subplots()
            colors = list(mcolors.TABLEAU_COLORS.values())[:len(summary)]
            ax.pie(summary.values, labels=summary.index, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
