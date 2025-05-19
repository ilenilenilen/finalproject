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
        sentences.extend(re.split(r"[\n•-]+", s))
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

# --- STREAMLIT APP ---

st.title("📄 CV Parsing and Text Classification")
st.markdown("Easily extract and categorize text from CVs to identify education, experience, requirements, responsibilities, skills, and soft skills.")

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

        # Convert categorized sentences to DataFrame
        df_categorized = pd.DataFrame(categorized)
        df_categorized.index += 1  # Change index to start from 1

        # Highlight categories in the DataFrame
        def highlight_categories(row):
            colors = {
                "Education": "#FFDDC1",
                "Experience": "#FFC1C1",
                "Requirement": "#C1E1FF",
                "Responsibility": "#FFDAC1",
                "Skill": "#C1FFC1",
                "SoftSkill": "#C1C1FF",
            }
            category_color = colors.get(row["category"], "#FFFFFF")
            return [f"background-color: {category_color};"] * len(row)

        st.dataframe(df_categorized.style.apply(highlight_categories, axis=1, subset=["category", "text"]))

        # Generate category distribution and summary
        all_categories = [item['category'] for item in categorized]
        df_cat = pd.Series(all_categories).value_counts()

        # Display summary with colors
        st.markdown("### Summary")
        for i, (cat, count) in enumerate(df_cat.items(), start=1):
            color = mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS.keys())[i % len(mcolors.TABLEAU_COLORS)]]
            st.markdown(f"""
                <div style="background-color: {color}; border-radius: 10px; padding: 10px; margin-bottom: 10px; color: white;">
                    <strong>{i}. {cat}</strong>: {count}
                </div>
            """, unsafe_allow_html=True)

        # Pie chart for category distribution
        st.subheader("Category Distribution (Pie Chart)")
        fig, ax = plt.subplots()
        colors = list(mcolors.TABLEAU_COLORS.values())[:len(df_cat)]
        ax.pie(df_cat.values, labels=df_cat.index, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
