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
                categorized_sentences.append({"text": sent, "category": cat})

    return categorized_sentences

# STREAMLIT APP

st.title("📄 CV Parsing with Job Description Manual Matching")

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

job_text_combined = (job_desc + " " + job_qual).strip()

st.subheader("Text Input for Classification")
text_for_classification = st.text_area(
    "Enter CV text manually or use extracted CV text above:",
    value=cv_text if cv_text.strip() else "",
    height=200,
)

model_choice = st.selectbox("Choose a model:", list(models.keys()))

if st.button("Predict"):
    if not text_for_classification.strip():
        st.warning("Please enter or select some CV text for classification.")
    else:
        model = models[model_choice]
        prediction = model.predict([text_for_classification])[0]
        st.success(f"Prediction: {prediction}")

        categorized = categorize_sentences(text_for_classification)

        # Buat dataframe dengan kolom manual_match yang bisa diedit checkbox-nya
        df_categorized = pd.DataFrame(categorized)
        if df_categorized.empty:
            st.info("No categorized sentences from CV found.")
        else:
            df_categorized.index += 1

            st.subheader("Manual Match Sentences with Job Description and Qualifications")
            st.markdown("Centang kalimat yang menurut Anda relevan dengan Job Description dan Kualifikasi:")

            # Buat list untuk checkbox manual match
            manual_matches = []
            for i, row in df_categorized.iterrows():
                checked = st.checkbox(f"[{row['category']}] {row['text']}", key=f"match_{i}")
                manual_matches.append(checked)

            # Tambahkan kolom match hasil manual ke df
            df_categorized["match_with_job_desc"] = manual_matches

            # Tampilkan dataframe dengan highlight berdasarkan manual match
            def highlight_categories(row):
                colors = {
                    "Education": "#FFDDC1",
                    "Experience": "#FFC1C1",
                    "Requirement": "#C1E1FF",
                    "Responsibility": "#FFDAC1",
                    "Skill": "#C1FFC1",
                    "SoftSkill": "#C1C1FF",
                }
                base_color = colors.get(row["category"], "#FFFFFF")
                if row["match_with_job_desc"]:
                    return [f"background-color: #B2FFB2;"] * len(row)
                else:
                    return [f"background-color: {base_color};"] * len(row)

            st.subheader("CV Categorization with Manual Matching Result")
            st.dataframe(df_categorized.style.apply(highlight_categories, axis=1))

            # Summary kategori hanya yang dicentang manual match
            df_matched = df_categorized[df_categorized["match_with_job_desc"] == True]
            df_cat = df_matched["category"].value_counts()

            st.markdown("### Summary of CV Categories Matching Job Description (Manual)")
            if not df_cat.empty:
                for i, (cat, count) in enumerate(df_cat.items(), start=1):
                    color = mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS.keys())[i % len(mcolors.TABLEAU_COLORS)]]
                    st.markdown(f"""
                        <div style="background-color: {color}; border-radius: 10px; padding: 10px; margin-bottom: 10px; color: white;">
                            <strong>{i}. {cat}</strong>: {count}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No sentences manually matched with Job Description and Qualifications.")

            st.subheader("Category Distribution (Pie Chart) for Manually Matched CV Text")
            if not df_cat.empty:
                fig, ax = plt.subplots()
                colors = list(mcolors.TABLEAU_COLORS.values())[:len(df_cat)]
                ax.pie(df_cat.values, labels=df_cat.index, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            else:
                st.info("No data to display in pie chart.")

            # Tampilkan Job Description dan Kualifikasi juga di bawah untuk referensi
            st.subheader("Job Description and Qualifications Reference")
            st.markdown("**Job Description:**")
            st.write(job_desc)
            st.markdown("**Job Qualifications:**")
            st.write(job_qual)

            # Download button untuk Excel export
            def to_excel(df, job_desc, job_qual):
                output = pd.ExcelWriter("CV_Parsed_Results.xlsx", engine='xlsxwriter')
                df.to_excel(output, index=False, sheet_name='Categorized Sentences')

                # Tambahkan sheet job description & qualification
                workbook = output.book
                ws = workbook.add_worksheet('Job Description & Qualification')
                output.sheets['Job Description & Qualification'] = ws

                ws.write(0, 0, "Job Description")
                ws.write(1, 0, job_desc)
                ws.write(3, 0, "Job Qualifications")
                ws.write(4, 0, job_qual)

                output.close()
                with open("CV_Parsed_Results.xlsx", "rb") as f:
                    data = f.read()
                return data

            excel_data = to_excel(df_categorized, job_desc, job_qual)
            st.download_button(
                label="Download Results as Excel",
                data=excel_data,
                file_name="CV_Parsed_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
