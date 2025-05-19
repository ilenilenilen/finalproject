# Import libraries
import re
import os
import joblib
import PyPDF2
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
        "SVM": joblib.load(os.path
