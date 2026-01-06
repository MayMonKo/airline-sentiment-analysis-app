import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from scipy.sparse import hstack, csr_matrix

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load models
svm_model = joblib.load("model/svm_model.pkl")
tfidf = joblib.load("model/tfidf.pkl")
w2v_model = Word2Vec.load("model/w2v.model")

# Preprocessing setup
stop_words = set(stopwords.words("english")) - {"not", "no", "never"}
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 1
    ]
    return tokens

def embed_review(tokens, model):
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def predict_sentiment(text):
    tokens = preprocess(text)
    clean_text = " ".join(tokens)

    tfidf_vec = tfidf.transform([clean_text])
    w2v_vec = embed_review(tokens, w2v_model).reshape(1, -1)
    hybrid_vec = hstack([tfidf_vec, csr_matrix(w2v_vec)])

    return svm_model.predict(hybrid_vec)[0]

# Streamlit UI
st.set_page_config(page_title="Airline Review Sentiment Analyzer")

st.title("✈️ Airline Review Sentiment Analyzer")
st.write("Enter a customer review and get instant sentiment prediction.")

user_input = st.text_area("Enter airline review:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment.upper()}**")
    else:
        st.warning("Please enter a review.")

