---
title: Ford Sentence Classification
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
- nlp
- classification
pinned: true
short_description: An NLP sentence classifier with a simple web interface built using Streamlit and Docker
---

# 🚀 Ford Sentence Classification

**Ford Sentence Classification** is a lightweight, interactive web app for classifying textual sentences using a machine learning model — built with **Streamlit**, powered by **scikit-learn**, and deployed using **Docker**.

Whether you're testing NLP ideas or building a sentence classifier for real use, this app helps you do it quickly with an intuitive UI.

---

## 📘 What This App Does

This app:
- Accepts a sentence or phrase from user input
- Uses a trained ML model to classify the sentence into a predefined category
- Displays the prediction and optional probability score

Useful for:
- Prototyping NLP classification tools
- Demonstrating ML models with non-technical users
- Embedding into internal workflows or educational projects

---

## 🧩 Dataset / Database Column Reference

If your app uses a database or CSV with structured sentence data, here’s a sample breakdown (adjust as needed):

| Column Name       | Description                                        |
|-------------------|----------------------------------------------------|
| `id`              | Unique identifier for each record                  |
| `sentence`        | The input sentence or text to be classified        |
| `type `           | The true label/category                            |

You can adapt this based on your actual CSV or database structure.


## 🎯 Features

- ✨ Simple and clean Streamlit UI
- 🧠 Fast sentence classification
- 📦 Dockerized for easy deployment anywhere
- 📈 Extendable with your own model and data
- 🔍 Option to show prediction confidence

---

## 📦 Requirements

pandas
streamlit
joblib
numpy
scikit-learn
PyPDF2
matplotlib
nltk
xlsxwriter
streamlit-aggrid
