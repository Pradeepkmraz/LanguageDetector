import streamlit as st
import pickle

# Set up the web page styling
st.set_page_config(page_title="Language Detector", page_icon="🌐", layout="centered")
st.title("🌐 ZED-F Language Detection Model")
st.write("Type any sentence below, and the model will predict its language!")

# 1. Load the saved model and vectorizer safely
@st.cache_resource
def load_assets():
    with open("language_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        cv = pickle.load(f)
    return model, cv

try:
    model, cv = load_assets()
except Exception as e:
    st.error("Error loading model files. Make sure .pkl files are in the same folder.")

# 2. Build the user interface text input
user_input = st.text_area("Enter text here:", placeholder="e.g., My name is Pradeep Yadav")

if st.button("Predict Language"):
    if user_input.strip() == "":
        st.warning("Please type something first!")
    else:
        # Transform and predict using your pipeline steps
        vectorized_data = cv.transform([user_input]).toarray()
        
        # Check if the text actually matched the vocabulary
        if vectorized_data.sum() == 0:
            st.info("Notice: The input contains words completely new to the model's vocabulary.")
            
        prediction = model.predict(vectorized_data)
        
        # Display results cleanly
        st.success(f"🎉 Predicted Language: **{prediction[0]}**")
