# --- Tambahan Library ---
import os
import io
import re
import joblib
import PyPDF2
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from st_aggrid import AgGrid, GridOptionsBuilder

# Download tokenizer
nltk.download("punkt", quiet=True)
tokenizer = PunktSentenceTokenizer()

# Directory for model files
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl")),
        # "Ensemble": joblib.load(os.path.join(MODEL_DIR, "ensemble_model.pkl")),
        # "SVM": joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl")),
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
    sentences = tokenizer.tokenize(text)
    return [s.strip() for s in sentences if s.strip()]

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Manual Matching")
    return output.getvalue()

# -------------------- STREAMLIT APP --------------------
st.title("📄 CV Parsing and Job Description Matching (with AgGrid)")
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
job_desc = st.text_area("Enter Job Description:", height=150)

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
        sentences = categorize_sentences(text_for_classification)

        predictions = [model.predict([sentence])[0] for sentence in sentences]
        df_results = pd.DataFrame({
            "Sentence": sentences,
            "Prediction": predictions,
            "Manual Match": [False] * len(sentences)
        })

        st.subheader("Categorized Sentences with Predictions (Editable)")
        gb = GridOptionsBuilder.from_dataframe(df_results)
        gb.configure_column("Manual Match", editable=True, cellEditor='agCheckboxCellEditor')
        gb.configure_column("Sentence", wrapText=True, autoHeight=True)
        gb.configure_grid_options(domLayout='normal')

        grid_options = gb.build()
        grid_response = AgGrid(
            df_results,
            gridOptions=grid_options,
            enable_enterprise_modules=False,
            fit_columns_on_grid_load=True,
            update_mode='VALUE_CHANGED',
            allow_unsafe_jscode=True,
            height=400,
        )

        edited_df = grid_response['data']
        df_results = pd.DataFrame(edited_df)

        st.subheader("Summary of Predictions")
        df_cat = df_results["Prediction"].value_counts()
        for i, (cat, count) in enumerate(df_cat.items(), start=1):
            color = mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS.keys())[i % len(mcolors.TABLEAU_COLORS)]]
            st.markdown(
                f"<div style='background-color: {color}; padding: 10px; margin: 5px; color: white; border-radius: 5px;'>"
                f"<strong>{i}. {cat}</strong>: {count}</div>",
                unsafe_allow_html=True,
            )

        st.subheader("Prediction Distribution (Pie Chart)")
        fig, ax = plt.subplots()
        colors = list(mcolors.TABLEAU_COLORS.values())[:len(df_cat)]
        ax.pie(df_cat.values, labels=df_cat.index, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        # Download Excel
        st.download_button(
            label="📥 Download Predictions + Manual Match (Excel)",
            data=to_excel(df_results),
            file_name="cv_match_aggrid.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
