import re
import os
import joblib
import PyPDF2
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

nltk.download('punkt', quiet=True)
tokenizer = PunktSentenceTokenizer()

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

categories = {
    "Education": ["education", "degree", "university", "bachelor", "master", "phd", "gpa"],
    "Experience": [
        "experience", "worked", "job", "position", "years", "intern", "internship",
        "architect", "staff", "junior", "developed", "created", "produced", "coordinated",
        "managed", "designed"
    ],
    "Requirement": [
        "requirement", "mandatory", "qualification", "criteria", "must", "eligibility", 
        "prerequisite", "need", "essential", "necessary"
    ],
    "Responsibility": ["responsibility", "task", "duty", "role", "accountable", "responsible"],
    "Skill": [
        "skill", "expertise", "proficiency", "tools", "excel", "project management", 
        "research", "problem solving", "public speaking", "technical", "coding", "programming"
    ],
    "SoftSkill": [
        "communication", "leadership", "teamwork", "problem-solving", "advocacy", 
        "relationship building", "empathy", "negotiation", "interpersonal", 
        "creativity", "adaptability"
    ],
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
    raw_sentences = tokenizer.tokenize(text)
    
    sentences = []
    for s in raw_sentences:
        parts = re.split(r"[\n•\-●]+", s)
        parts = [p.strip() for p in parts if p.strip()]
        sentences.extend(parts)

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

# Streamlit app
st.title("CV Parsing and Text Classification")

uploaded_file = st.file_uploader("Upload your CV (PDF format only):", type=["pdf"])

cv_text = ""

if uploaded_file is not None:
    cv_text = extract_text_from_pdf(uploaded_file)
    if cv_text.strip():
        st.text_area("Extracted Text from PDF:", cv_text, height=200)
    else:
        st.error("No text could be extracted from the uploaded file.")

st.subheader("Or Enter Text Manually for Classification")
text = st.text_area("Paste text here:", value=cv_text if cv_text.strip() else "", height=200)

if st.button("Categorize Sentences"):
    if not text.strip():
        st.warning("Please upload a PDF or enter some text first.")
    else:
        categorized = categorize_sentences(text)
        if not categorized:
            st.info("No categories found.")
        else:
            df_cat = pd.DataFrame(categorized)
            df_cat.index = df_cat.index + 1  # start index at 1
            
            st.dataframe(df_cat.rename(columns={"text":"Text", "category":"Category"}), use_container_width=True)

            # Summary category counts
            st.markdown("### Summary Category Counts")
            summary_counts = df_cat['category'].value_counts()

            color_map = {
                "Education": "blue",
                "Experience": "green",
                "Requirement": "orange",
                "Responsibility": "purple",
                "Skill": "red",
                "SoftSkill": "brown"
            }

            for i, (cat, count) in enumerate(summary_counts.items(), start=1):
                color = color_map.get(cat, "black")
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>{i}. {cat}: {count}</span>", unsafe_allow_html=True)

            # Pie chart for category distribution
            fig, ax = plt.subplots()
            colors = [color_map.get(cat, "grey") for cat in summary_counts.index]
            ax.pie(summary_counts.values, labels=summary_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.pyplot(fig)
