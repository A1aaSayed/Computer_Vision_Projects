import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

model = joblib.load('fake_news_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


st.title('📰 Fake News Detection App')

input_text = st.text_area('Enter News Article or Headline')

if st.button('Predict'):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess_text(input_text)
        transformed = tfidf.transform([cleaned])
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0]
        confidence = round(max(probability) * 100, 2)

        # label = '🟢 Real News' if prediction == 1 else '🔴 Fake News'
        # st.success(f'Prediction: {label} ({confidence}% confidence)')

        st.subheader("🔍 Prediction Result")
        if prediction == 1:
            st.markdown(
                f"""
                <div style="background-color:#d4edda;padding:20px;border-radius:10px;">
                    <h3 style="color:#155724;">🟢 Real News</h3>
                    <p style="color:#155724;">Confidence: <b>{confidence}%</b></p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color:#f8d7da;padding:20px;border-radius:10px;">
                    <h3 style="color:#721c24;">🔴 Fake News</h3>
                    <p style="color:#721c24;">Confidence: <b>{confidence}%</b></p>
                </div>
                """, unsafe_allow_html=True
            )
