# Import libraries
import re
import os
import io
import joblib
import PyPDF2
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

# Download punkt tokenizer if not available
nltk.download('punkt', quiet=True)

# Initialize Punkt tokenizer
tokenizer = PunktSentenceTokenizer()

# Replace with your model directory path if needed
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl")),
        "Ensemble": joblib.load(os.path.join(MODEL_DIR, "ensemble_model.pkl"), mmap_mode='r'),
        "SVM": joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl")),
    }

# Extract text from uploaded PDF file
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

# Categorize sentences
def categorize_sentences(text):
    categories = {
        "Education": ["education", "degree", "university", "bachelor", "master", "phd", "gpa",
                      "computer science", "mathematics", "statistics", "information system", "relevant major"],
        "Experience": ["experience", "worked", "job", "position", "years", "intern", "5 years",
                       "data science", "data scientist", "fintech", "finance services", "production environment"],
        "Requirement": ["requirement", "mandatory", "qualification", "criteria", "must", "eligibility",
                        "deep understanding", "strong analytical thinking", "proven experience", "advantage"],
        "Responsibility": ["responsibility", "task", "duty", "role", "accountable", "responsible",
                           "design", "build", "deploy", "perform testing", "model implementation", "fine tuning", "drive improvement"],
        "Skill": ["skill", "expertise", "proficiency", "tools", "excel", "project management", "research",
                  "problem solving", "public speaking", "machine learning", "model development", "model deployment",
                  "risk evaluation", "business impact analysis", "feature engineering", "algorithm", "analysis"],
        "SoftSkill": ["communication", "leadership", "teamwork", "problem-solving", "advocacy",
                      "relationship building", "analytical thinking"]
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

# Streamlit App
st.title("📄 CV Parsing with Manual Job Description Matching")

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

st.subheader("Job Description Input (Optional)")
st.text_area("Enter Job Description (not used for matching, only reference):", height=150)
st.text_area("Enter Job Qualifications:", height=150)

st.subheader("Text Input for Classification")
text_for_classification = st.text_area(
    "Enter CV text manually or use extracted CV text above:",
    value=cv_text if cv_text.strip() else "",
    height=200,
)

model_choice = st.selectbox("Choose a model:", list(models.keys()))

if st.button("Predict and Categorize"):
    if not text_for_classification.strip():
        st.warning("Please enter or select some CV text for classification.")
    else:
        model = models[model_choice]
        prediction = model.predict([text_for_classification])[0]
        st.success(f"Prediction: {prediction}")

        categorized = categorize_sentences(text_for_classification)

        if categorized:
            df_categorized = pd.DataFrame(categorized)
            df_categorized.index += 1
            df_categorized["match_with_job_desc"] = False  # Manual match default

            st.subheader("CV Categorization Matching Job Description (Manual Matching)")
            edited_df = st.data_editor(
                df_categorized,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "match_with_job_desc": st.column_config.CheckboxColumn("Match with Job Description")
                }
            )

            # Summary only for matched rows
            matched_df = edited_df[edited_df["match_with_job_desc"] == True]
            df_cat = matched_df["category"].value_counts()

            if not matched_df.empty:
                st.markdown("### Summary of CV Categories Matching Job Description")
                for i, (cat, count) in enumerate(df_cat.items(), start=1):
                    color = mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS.keys())[i % len(mcolors.TABLEAU_COLORS)]]
                    st.markdown(f"""
                        <div style="background-color: {color}; border-radius: 10px; padding: 10px; margin-bottom: 10px; color: white;">
                            <strong>{i}. {cat}</strong>: {count}
                        </div>
                    """, unsafe_allow_html=True)

                st.subheader("Category Distribution (Pie Chart) for CV Text Matching Job Description")
                fig, ax = plt.subplots()
                colors = list(mcolors.TABLEAU_COLORS.values())[:len(df_cat)]
                ax.pie(df_cat.values, labels=df_cat.index, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

            # ✅ FIXED: Save Excel without calling `.save()`
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                edited_df.to_excel(writer, index=True, sheet_name="CV_Match")
            output.seek(0)  # Move cursor to start of the stream

            st.download_button(
                label="📥 Download Matched CV Data as Excel",
                data=output.getvalue(),
                file_name="cv_match_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("No categorized sentences found in the provided CV text.")
