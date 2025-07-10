# 🧠 MCQ Difficulty Estimator

A smart AI-powered web application that predicts the **difficulty level** of multiple-choice questions (MCQs) using a fine-tuned BERT model.

## 🚀 Overview

This project allows users (students, educators, or e-learning platforms) to:

- Upload a CSV or Excel file containing MCQs.
- Automatically classify each question as:
  - 🟢 Easy
  - 🟡 Medium
  - 🔴 Hard
  - 🔵 Very Hard
- Get interactive feedback, scores, time tracking, and performance analysis.
- Export the sorted questions.

## 🚀 Demo

Here's a short demo of the app in action:

![MCQ Difficulty Estimator Demo](demo.gif)

## 🛠️ Tech Stack

| Component     | Details                                                                 |
|---------------|-------------------------------------------------------------------------|
| 🧠 Model       | BERT (`bert-base-uncased`) fine-tuned on labeled MCQs                  |
| 🧪 Framework   | `transformers`, `datasets`, `evaluate`, `scikit-learn`                 |
| 📊 App         | `Streamlit` with interactive UI & visualizations (`plotly`)           |
| 📝 Dataset     | Custom dataset with `question`, `options`, `answer`, and `difficulty` |

## 📂 Sample Input Format

The app expects the following columns in your uploaded CSV/Excel file:

question | choiceA | choiceB | choiceC | choiceD | answerKey


> Example:
>
> `What is the capital of France? | Paris | London | Rome | Berlin | A`

## 🧠 Model Details

The model is fine-tuned on a labeled dataset of MCQs using the Hugging Face 🤗 `transformers` library. It predicts the difficulty level based on question phrasing, distractor quality, and answer choices.

| Label        | Encoding |
|--------------|----------|
| Easy         | 0        |
| Hard         | 1        |
| Medium       | 2        |
| Very_Hard    | 3        |


## 📦 Installation

```bash
git clone https://github.com/A1aaSayed/AI_Projects.git
cd mcq_difficulty_estimator
pip install -r requirements.txt
streamlit run app.py
```

## 📈 Future Improvements
Add multilingual support (e.g., Arabic, French).

Improve adaptive learning strategy.

Integrate with a user database for saving performance.