import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load trained TF-IDF Vectorizer and ML Model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Preprocessing Function
def transform_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "url", text)  # Replace URLs with 'url'
    words = nltk.word_tokenize(text)
    filtered_words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

# Streamlit UI Setup
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("This app uses a trained ML model to detect whether a message is spam or not.")

input_sms = st.text_area("📨 Enter the message below", height=150)

# When button is clicked
if st.button('🔍 Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # 1. Transform the text
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms]).toarray()

        # 3. Predict label
        result = model.predict(vector_input)[0]

        # 4. Predict confidence (if available)
        confidence = None
        if hasattr(model, 'predict_proba'):
            confidence = round(model.predict_proba(vector_input)[0][result] * 100, 2)

        # 5. Output result
        st.markdown("---")
        if result == 1:
            st.markdown("<h2 style='color: red;'>🚫 Spam Detected!</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: green;'>✅ Not Spam</h2>", unsafe_allow_html=True)

        if confidence is not None:
            st.markdown(f"**Confidence:** {confidence}%")

        with st.expander("🧪 View Processed Message"):
            st.code(transformed_sms)
