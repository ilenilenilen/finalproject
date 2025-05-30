#Import Library
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
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

#Download model tokenizer
nltk.download("punkt", quiet=True)
#Membuat instance tokenizer untuk teks
tokenizer = PunktSentenceTokenizer()

#Mendapatkan path direktori skrip untuk memuat model
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

#Cache Models
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl")),
    }

#Extract text from PDF
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

#sentence categorization sentence
def categorize_sentences(text):
    #Membagi teks menjadi kalimat
    sentences = tokenizer.tokenize(text)
    #Menghapus kalimat kosong
    return [s.strip() for s in sentences if s.strip()]

#Dataframe to excel function
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Manual Matching")
    return output.getvalue()

# --- Streamlit App ---
st.title("📄 Job Description Classification Text")

models = load_models()

#Mengunggah file
uploaded_file = st.file_uploader("Upload your CV (PDF format only):", type=["pdf"])
cv_text = ""
if uploaded_file is not None:
    with uploaded_file:
        cv_text = extract_text_from_pdf(uploaded_file)
    if cv_text.strip():
        st.text_area("Extracted CV Text:", cv_text, height=200)
    else:
        st.error("No text could be extracted from the uploaded file.")

#Job description
st.subheader("Job Description Input")
job_desc = st.text_area("Enter Job Description:", height=150)

st.subheader("Text Input for Classification")
text_for_classification = st.text_area(
    "Enter CV text manually or use extracted CV text above:",
    value=cv_text if cv_text.strip() else "",
    height=200,
)

model_choice = st.selectbox("Choose a model:", list(models.keys()))

#Initialize session state
if "df_results" not in st.session_state:
    st.session_state.df_results = None

if st.button("Predict"):
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

        st.session_state.df_results = df_results

#AgGrid and ouput display
if st.session_state.df_results is not None:
    st.subheader("Categorized Sentences with Predictions (Editable)")
    gb = GridOptionsBuilder.from_dataframe(st.session_state.df_results)
    gb.configure_column("Manual Match", editable=True, cellEditor='agCheckboxCellEditor')
    gb.configure_column("Sentence", wrapText=True, autoHeight=True)
    gb.configure_grid_options(domLayout='normal')
    grid_options = gb.build()

    grid_response = AgGrid(
        st.session_state.df_results,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True,
        height=400,
    )

    if grid_response['data'] is not None:
        st.session_state.df_results = pd.DataFrame(grid_response['data'])

    #Summary
    df_cat = st.session_state.df_results["Prediction"].value_counts()
    st.subheader("Summary of Predictions")
    for i, (cat, count) in enumerate(df_cat.items(), start=1):
        color = mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS.keys())[i % len(mcolors.TABLEAU_COLORS)]]
        st.markdown(
            f"<div style='background-color: {color}; padding: 10px; margin: 5px; color: white; border-radius: 5px;'>"
            f"<strong>{i}. {cat}</strong>: {count}</div>",
            unsafe_allow_html=True,
        )

    #Pie chart
    st.subheader("Prediction Distribution (Pie Chart)")
    fig, ax = plt.subplots()
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(df_cat)]
    ax.pie(df_cat.values, labels=df_cat.index, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    #Excel download
    st.download_button(
        label="📥 Download Predictions + Manual Match (Excel)",
        data=to_excel(st.session_state.df_results),
        file_name="cv_match_aggrid.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
