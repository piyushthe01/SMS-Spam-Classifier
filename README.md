# 📩 SMS/Email Spam Classifier

This repository contains a complete machine learning pipeline for classifying SMS and email messages as **spam** or **ham** (not spam). It includes exploratory data analysis, model training, and deployment using Streamlit.
---

## 🧠 Project Overview

### ✅ Notebook Highlights

The notebook performs:

1. **Data Cleaning**  
   - Dropped unnecessary columns  
   - Renamed columns for clarity

2. **Exploratory Data Analysis (EDA)**  
   - Class imbalance visualization  
   - WordCloud and bar charts of frequent words  
   - Found: Spam messages tend to be longer

3. **Text Preprocessing**  
   - Lowercasing, stopword removal, stemming  
   - Custom tokenizer using `nltk` and `PorterStemmer`

4. **Feature Engineering**  
   - Bag of Words  
   - TF-IDF Vectorization

5. **Model Building & Evaluation**  
   - Trained **Multinomial Naive Bayes (MNB)** on TF-IDF vectors  
   - Evaluated using precision, recall, and F1-score  
   - Tried ensemble models like Voting and Stacking Classifier for improvement

6. **Model Export**  
   - Saved `model.pkl` and `vectorizer.pkl` for deployment

---

## 💻 App Features (Streamlit)

- ✅ Classify message as Spam or Not Spam
- 📊 Display prediction confidence (if available)
- 🔍 View processed (cleaned) message text
- 🧼 Simple and clean interface

---

## 📂 Repository Structure

├── streamlit_app.py # Streamlit web app
├── model.pkl # Trained classifier
├── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Python dependencies
├── SMS Spam Classifiier.ipynb # Full EDA + Model training notebook
└── README.md # Project info

---

## 📊 Dataset Used

- 📁 **SMS Spam Collection Dataset**  
  🔗 [View on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🛠 Technologies Used

- Python
- NLTK
- Scikit-learn
- Streamlit
- Pandas, Matplotlib, Seaborn

---

## 👤 Author

**Piyush Mishra**  
📧 [piyushmishra27j@gmail.com](mailto:piyushmishra27j@gmail.com)  
🔗 [LinkedIn - mishrapm](https://www.linkedin.com/in/mishrapm)
