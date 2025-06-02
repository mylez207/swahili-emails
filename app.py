# app.py

import streamlit as st
import joblib
import re
import string

# Load your pre-trained model and vectorizer
model = joblib.load("swahili_sms_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Make sure this exists!

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Streamlit UI
st.title("📧 Swahili Email Spam Detector")
st.markdown("Enter a Swahili message below and check if it's **SCAM** or **TRUST**.")

email_input = st.text_area("✉️ Message", height=150)

if st.button("🔍 Analyze"):
    if email_input.strip():
        cleaned = preprocess_text(email_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction.lower() == "scam":
            st.error("🚨 This message is classified as: **SCAM**")
        else:
            st.success("✅ This message is classified as: **TRUST**")
    else:
        st.warning("Please enter a message first.")
