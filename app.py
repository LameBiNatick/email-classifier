import streamlit as st
import joblib
import numpy as np


# -----------------------------
# Load Saved Model
# -----------------------------
def load_model(prefix='email_classifier'):
    nb = joblib.load(f'{prefix}_nb.joblib')
    tfidf = joblib.load(f'{prefix}_tfidf.joblib')
    label_encoder = joblib.load(f'{prefix}_label_encoder.joblib')
    return nb, tfidf, label_encoder


# -----------------------------
# Streamlit App
# -----------------------------
def run_streamlit_app():
    st.title("📧 Email Classifier")
    st.write("Classifies emails into: Spam, Promotions, Updates, Personal")

    nb, tfidf, label_encoder = load_model()

    email_text = st.text_area("Paste your email text here:")

    if st.button("Classify"):
        if email_text.strip() == "":
            st.warning("Please enter some email text.")
        else:
            features = tfidf.transform([email_text])
            pred_idx = nb.predict(features)[0]
            pred_class = label_encoder.inverse_transform([pred_idx])[0]
            probas = nb.predict_proba(features)[0]

            st.subheader(f"Prediction: {pred_class}")
            prob_dict = {cls: float(prob) for cls, prob in zip(label_encoder.classes_, probas)}
            st.write("Probability scores:", prob_dict)

            st.bar_chart(np.array([probas]), use_container_width=True)


if __name__ == "__main__":
    run_streamlit_app()
