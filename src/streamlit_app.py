import re
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('punkt', quiet=True)
tokenizer = PunktSentenceTokenizer()

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

def categorize_sentences(text):
    # Tokenize into sentences first
    raw_sentences = tokenizer.tokenize(text)
    
    sentences = []
    # Split each sentence by line breaks and bullets, including ● and others
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
        
        # Debug print (hapus atau comment kalau sudah oke)
        # if matched_category == "Experience":
        #     print(f"[DEBUG Experience] {sent}")
        
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


# --- Contoh sederhana streamlit app ---
st.title("CV Parsing and Text Classification")

text = st.text_area("Paste CV text here:")

if st.button("Categorize Sentences"):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        categorized = categorize_sentences(text)
        if not categorized:
            st.info("No categories found.")
        else:
            df_cat = pd.DataFrame(categorized)
            df_cat.index = df_cat.index + 1  # start index at 1
            
            # Tampilkan tabel dengan kategori dan teks
            st.dataframe(df_cat.rename(columns={"text":"Text", "category":"Category"}))

            # Summary tanpa index
            st.markdown("### Summary Category Counts")
            summary_counts = df_cat['category'].value_counts()
            for i, (cat, count) in enumerate(summary_counts.items(), start=1):
                color = {
                    "Education": "blue",
                    "Experience": "green",
                    "Requirement": "orange",
                    "Responsibility": "purple",
                    "Skill": "red",
                    "SoftSkill": "brown"
                }.get(cat, "black")
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>{i}. {cat}: {count}</span>", unsafe_allow_html=True)

            # Pie chart
            fig, ax = plt.subplots()
            ax.pie(summary_counts.values, labels=summary_counts.index, autopct='%1.1f%%', startangle=90,
                   colors=[
                       "blue", "green", "orange", "purple", "red", "brown"
                   ][:len(summary_counts)])
            ax.axis('equal')
            st.pyplot(fig)
