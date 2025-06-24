# 📰 Fake News Detection App

A web-based machine learning application that detects whether a news article is **Fake** or **Real** using Natural Language Processing and Logistic Regression.

This project was built using Python, Streamlit, and scikit-learn, and is trained on a real-world dataset of fake and true news articles.

---

## 🚀 Features

- 🔤 Text preprocessing (lowercasing, punctuation and stopwords removal)
- ✨ TF-IDF vectorization (with n-grams)
- 🤖 Logistic Regression model for classification
- 📊 Confidence score (probability) shown to the user
- 🧠 Pre-trained model (`.pkl` file) for fast predictions
- 🌐 Web interface built with Streamlit

---

## 🚀 Demo

Here's a short demo of the app in action:

![Fake Detection App Demo](demo.gif)

---

## 🧠 Model Training Overview

- Merged and labeled real and fake news datasets
- Applied text cleaning: lowercasing, removing digits/punctuation/stopwords
- Created a combined feature (`title` + `text`)
- TF-IDF vectorized with `max_features=5000` and bi-grams
- Trained a logistic regression classifier
- Achieved high accuracy and F1-score on validation set

---

## 🖥️ How to Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/A1aaSayed/Computer_Vision_Projects.git
cd fake-news-detector
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run fake_news_app.py
```

## 🧪 Sample Input
Paste any news article or headline below and the app will analyze and classify it as **🟢 Real** or **🔴 Fake**, along with a **confidence percentage**.

**Example:**

> 📝 _"President signs executive order to tackle climate change and reduce carbon emissions."_

**Predicted Output:**

> ✅ **Real News** (Confidence: 92.3%)

## 📚 Dataset Source
This project uses the **[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)** available on Kaggle.