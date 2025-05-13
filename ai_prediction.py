import streamlit as st
import pandas as pd
import joblib
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="AI Project Demo", layout="centered")

# ---- Sidebar ----
st.sidebar.title("🔍 AI Tool Settings")
st.sidebar.markdown("Upload your model or input your text below")

# ---- Title ----
st.title("🤖 AI-Powered Prediction App")
st.markdown("Upload your trained ML model and try predictions instantly.")

# ---- Model Upload ----
model_file = st.sidebar.file_uploader("📁 Upload your model (.pkl)", type=["pkl"])

# ---- Input Area ----
text_input = st.text_area("✍️ Enter your input text/data here", height=150)

# ---- Predict Button ----
if st.button("🔮 Predict"):
    if model_file and text_input:
        # Load model
        model = joblib.load(model_file)
        vectorizer = CountVectorizer()
        input_vector = vectorizer.fit_transform([text_input])

        try:
            prediction = model.predict(input_vector)
            st.success(f"✅ **Prediction:** {prediction[0]}")
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")
    else:
        st.warning("⚠️ Please upload a model and enter text.")

# ---- Optional Suggestions ----
if st.button("💡 Suggest Improvements (AI)"):
    st.write("🔧 AI Suggestion Example:")
    st.info("Try including more relevant keywords or metrics to boost model confidence.")
