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
nltk.download ('punk', quiet = True)

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

#Extract file from PDF
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
 
def check_match(sentence, job_text):
    """Check if a sentence contains words or phrases relevant to the job description text."""
    sent_lower = sentence.lower()
    job_lower = job_text.lower()
    #Check if at least one word in the sentence exists in thejob description text
    return any(word in job_lower for word in sent_lower.split())

def categorize_sentences(text):
    #Define categories and their corresponding keywords
    categories ={
        "Education":[
            "education","degree","university","bachelor","master","phd","gpa",
            "computer science","mathematics","statistics","information system","relevant major"
        ],
        "Experience":[
            "experience","worked","job","position","years","intern",
            "5 years","data science","data scientist","fintech","finance services",
            "production environment"
        ],
        "Requirement":[
            "requirement", "mandatory", "qualification", "criteria", "must", "eligibility",
            "deep understanding", "strong analytical thinking", "proven experience", "advantage"
        ],
        "Responsibility":[
            "responsibility","task","duty","role","accountable","responsible",
            "design","build","deploy","perform testing","model implementation",
            "fine tuning","drive improvement"
        ],
        "Skill":[
            "skill","expertise","proficiency","tools","excel",
            "project management","research","problem solving","public speaking",
            "machine learning","model development","model deployment","risk evaluation",
            "business impact analysis","feature engineering","algorithm","analysis"
        ],
        "SoftSkill": [
            "communication","leadership","teamwork","problem-solving","advocacy",
            "relationship building","analytical thinking"
        ],
    }

    #Tokenize the input text into individual sentences
    sentences = tokenizer.tokenize(text)

    #Remove loading and trailing whitespaces and filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    categorized_sentences = []

    for sent in sentences:
        sent_lower = sent.lower()
        matched_categories = []
        #Loop through each category and it's keywords
        for category, keywords in categories.items():
            #check if any keyword matches whole words in the sentence 
            if any(re.search(rf"\b{re.escape(kw)}\b", sent_lower) for kw in keywords):
                matched_categories.append(category)
        #Only save sentences that match at least one category (exclude uncategorized)
        if matched_categories:
            for cat in matched_categories:
                categorized_sentences.append({"text": sent, "category": cat})

    return categorized_sentences


#STREAMLIT APP

#Set the tittle of the Streamlit
st.title("📄 CV Parsing with Job Description Matching")

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

if st.button("Predict and Match"):
    if not text_for_classification.strip():
        st.warning("Please enter or select some CV text for classification.")
    else:
        model = models[model_choice]
        prediction = model.predict([text_for_classification])[0]
        st.success(f"Prediction: {prediction}")

        categorized = categorize_sentences(text_for_classification)

        # Filter kalimat yang match dengan job desc & qual saja
        if job_text_combined:
            filtered = []
            for item in categorized:
                if check_match(item["text"], job_text_combined):
                    filtered.append({**item, "match_with_job_desc": True})
        else:
            # Jika job desc kosong, tampilkan semua kategori tapi tanpa match
            filtered = [{**item, "match_with_job_desc": False} for item in categorized]

        df_categorized = pd.DataFrame(filtered)
        if df_categorized.empty:
            st.info("No categorized sentences from CV match the Job Description and Qualifications.")
        else:
            df_categorized.index += 1

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
                # Highlight hijau muda jika match
                if row["match_with_job_desc"]:
                    return [f"background-color: #B2FFB2;"] * len(row)
                else:
                    return [f"background-color: {base_color};"] * len(row)

            st.subheader("CV Categorization Matching Job Description")
            st.dataframe(df_categorized.style.apply(highlight_categories, axis=1, subset=["category", "text", "match_with_job_desc"]))

            # Summary kategori hanya kalimat yang match job desc & qual
            df_cat = df_categorized["category"].value_counts()

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
