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
    categories = {
        "Education": ["education", "degree", "university", "bachelor", "master", "phd", "gpa"],
        "Experience": ["experience", "worked", "job", "position", "years", "intern"],
        "Requirement": ["requirement", "mandatory", "qualification", "criteria", "must", "eligibility"],
        "Responsibility": ["responsibility", "task", "duty", "role", "accountable", "responsible"],
        "Skill": [
            "skill", "expertise", "proficiency", "tools", "excel",
            "project management", "research", "problem solving", "public speaking"
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

        match_results.append({
            "JD Category": jd_category,
            "JD Text": jd_row["text"],
            "Matches in CV": match_count
        })

    return pd.DataFrame(match_results)

def visualize_category_distribution(categories):
    category_counts = pd.Series([item['category'] for item in categories]).value_counts()

    st.subheader("Category Distribution")
    for i, (cat, count) in enumerate(category_counts.items(), start=1):
        color = mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS.keys())[i % len(mcolors.TABLEAU_COLORS)]]
        st.markdown(f"""
            <div style="background-color: {color}; border-radius: 10px; padding: 10px; margin-bottom: 10px; color: white;">
                <strong>{i}. {cat}</strong>: {count}
            </div>
        """, unsafe_allow_html=True)

    # Pie chart for category distribution
    st.subheader("Category Distribution (Pie Chart)")
    fig, ax = plt.subplots()
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(category_counts)]
    ax.pie(category_counts.values, labels=category_counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

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

model_choice = st.selectbox("Choose a model:", list(models.keys()))

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

        # Predict category if needed
        prediction_text = st.text_area("Enter text for model prediction:", value="")
        if prediction_text.strip():
            model = models[model_choice]
            prediction = model.predict([prediction_text])[0]
            st.success(f"Prediction: {prediction}")

        st.subheader("Categorized CV")
        cv_df = pd.DataFrame(cv_categories)
        st.dataframe(cv_df)

        visualize_category_distribution(cv_categories)

        st.subheader("Matching Results")
        st.dataframe(match_df.style.highlight_max(subset=["Matches in CV"], color="lightgreen", axis=0))
