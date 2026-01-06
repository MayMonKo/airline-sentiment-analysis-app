# ✈️ Airline Review Sentiment Analysis Web App

A deployed Natural Language Processing (NLP) application that classifies airline customer reviews into **Positive**, **Neutral**, or **Negative** sentiment using machine learning.

This project compares public sentiment between **Turkish Airlines** and **Qatar Airways** using real-world customer reviews and provides a live web interface for sentiment prediction.

---

## 🎯 Objective

- Perform supervised sentiment analysis on airline customer reviews
- Compare sentiment distributions between two competing airlines
- Evaluate multiple machine learning models
- Deploy the best-performing model as a real-time web application

---

## 📊 Dataset

- **Source:** Kaggle – Airline Reviews Dataset  
- **Time Range:** 2023  
- **Type:** Public, non-copyrighted  
- **Size:** 3,309 reviews  
- **Airlines Used:** Turkish Airlines, Qatar Airways  

Sentiment labels are derived from numerical ratings:
- 1–4 → Negative  
- 5–6 → Neutral  
- 7–10 → Positive  

---

## 🧠 Methodology

### Text Preprocessing
- Lowercasing
- URL and punctuation removal
- Tokenization
- Stopword removal with negation preservation
- Lemmatization

### Feature Engineering
- **TF-IDF (unigrams & bigrams)** for lexical sentiment cues
- **Word2Vec embeddings** for semantic representation
- **Hybrid TF-IDF + Word2Vec feature space**

### Machine Learning Models
- Logistic Regression (baseline)
- **Linear Support Vector Machine (primary model)**
- Random Forest (comparison)

Hyperparameter tuning and class weighting were applied.

---

## 📈 Evaluation

- Metrics: Accuracy, Precision, Recall, F1-score
- Per-class analysis revealed neutral sentiment as the most challenging
- Linear SVM achieved the highest overall performance and balanced classification

---

## 🌐 Web Application

The Streamlit web app allows users to:
- Enter a new airline review
- Receive instant sentiment prediction
- Test unseen reviews in real time

### Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
