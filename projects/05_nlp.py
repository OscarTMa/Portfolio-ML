import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- PAGE CONFIGURATION ---
st.markdown("## 🤖 NLP Sentiment Classifier")
st.markdown("""
This tool analyzes the **sentiment** (emotional tone) of a text using Natural Language Processing.
Enter a review, comment, or tweet to see if it's **Positive** or **Negative**.
""")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    path = 'models/nlp_sentiment_model.pkl'
    if os.path.exists(path):
        return joblib.load(path)
    return None

pipeline = load_model()

if pipeline is None:
    st.error("⚠️ Model not found! Please run the training notebook first.")
    st.stop()

# --- 2. USER INPUT ---
st.subheader("✍️ Enter Text")
user_text = st.text_area("Type your review here...", height=150, 
                         placeholder="Example: The service was amazing, but the food was cold.")

# --- 3. PREDICTION & VISUALIZATION ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Analyze Sentiment", type="primary"):
        if user_text.strip() == "":
            st.warning("Please enter some text first.")
        else:
            with st.spinner('Reading text...'):
                # Predict (returns 0 or 1)
                prediction = pipeline.predict([user_text])[0]
                probability = pipeline.predict_proba([user_text])[0] # [prob_neg, prob_pos]
                
                # Logic: 1 = Positive, 0 = Negative
                sentiment = "Positive" if prediction == 1 else "Negative"
                confidence = probability[1] if prediction == 1 else probability[0]
                
                # --- RESULTS ---
                st.markdown("### Result:")
                if sentiment == "Positive":
                    st.success(f"😊 **{sentiment} Sentiment**")
                    st.metric("Confidence Score", f"{confidence*100:.1f}%")
                else:
                    st.error(f"😠 **{sentiment} Sentiment**")
                    st.metric("Confidence Score", f"{confidence*100:.1f}%")
                    
                # Show raw probabilities breakdown
                st.caption(f"Probability breakdown: Negative: {probability[0]:.2f}, Positive: {probability[1]:.2f}")

with col2:
    # Word Cloud Visualization
    if user_text.strip() != "":
        st.markdown("### ☁️ Key Words")
        # Simple WordCloud generation
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(user_text)
        
        # Display using Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

# --- 4. EXPLANATION ---
st.markdown("---")
tab1, tab2 = st.tabs(["📘 How it Works", "🧠 TF-IDF Explained"])

with tab1:
    st.markdown("""
    ### The Pipeline
    1.  **Preprocessing:** The text is cleaned and standardized.
    2.  **Vectorization (TF-IDF):** Computers can't read words. We convert text into a matrix of numbers based on word frequency and importance.
    3.  **Classification:** A Logistic Regression model (trained on 50,000 movie reviews) calculates the probability of the text being positive.
    """)

with tab2:
    st.markdown("""
    ### What is TF-IDF?
    **Term Frequency - Inverse Document Frequency.**
    
    It's a statistical measure that evaluates how relevant a word is to a document in a collection of documents.
    * Words like "the", "is", "and" appear everywhere, so they get a **low score**.
    * Words like "fantastic", "terrible", "masterpiece" appear less often but carry more meaning, so they get a **high score**.
    """)
