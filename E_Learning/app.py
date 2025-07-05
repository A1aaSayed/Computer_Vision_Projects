import streamlit as st
st.set_page_config(page_title="MCQ Difficulty Estimator", layout="wide")

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import io
import time
import plotly.express as px

# ====== BERT Setup ======
@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model

tokenizer, model = load_bert()

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0].numpy()  # CLS token

# ====== Difficulty Scoring ======
def generate_paraphrases(question):
    return [
        question,
        f"What does this mean: {question}",
        f"How would you answer this: {question}",
        f"Can you solve this? {question}",
        f"Let's see if you can answer: {question}"
    ]

def compute_difficulty(df):
    df["full_text"] = df.apply(
        lambda row: f"{row['question']} [A] {row['choiceA']} [B] {row['choiceB']} [C] {row['choiceC']} [D] {row['choiceD']} [E] {row['choiceE']}",
        axis=1
    )

    embeddings = np.vstack(df["full_text"].apply(get_bert_embedding))
    norms = np.linalg.norm(embeddings, axis=1)

    scaler = MinMaxScaler(feature_range=(1, 10))
    df["difficulty_score"] = scaler.fit_transform(norms.reshape(-1, 1)).round(2)

    df["paraphrases"] = df["question"].apply(generate_paraphrases)

    df = df.sort_values("difficulty_score").reset_index(drop=True)
    return df

# ====== Streamlit UI ======
st.title("📚 Multiple Choice Question Difficulty Trainer")
st.markdown("Test your knowledge with adaptive question difficulty and paraphrasing!")

# ====== Session State Initialization ======
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "answered_correctly" not in st.session_state:
    st.session_state.answered_correctly = []
if "attempts_on_question" not in st.session_state:
    st.session_state.attempts_on_question = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "show_correct" not in st.session_state:
    st.session_state.show_correct = False
if "last_answer_time" not in st.session_state:
    st.session_state.last_answer_time = None
if "time_taken" not in st.session_state:
    st.session_state.time_taken = []

# ====== File Uploader ======
uploaded_file = st.file_uploader("📂 Upload your MCQ file (CSV or Excel)", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("❌ File must include columns: question, choiceA–E, answerKey")
        st.stop()

    required_cols = {"question", "choiceA", "choiceB", "choiceC", "choiceD", "choiceE", "answerKey"}
    if not required_cols.issubset(set(df.columns)):
        st.error("❌ File must include columns: question, choiceA–E, answerKey")
        st.stop()

    st.success("✅ File loaded successfully. Calculating difficulty...")
    result_df = compute_difficulty(df)

    # ====== Display Current Question ======
    current_index = st.session_state.current_question
    if current_index < len(result_df):
        row = result_df.iloc[current_index]
        paraphrases = row["paraphrases"]
        phrasing_index = st.session_state.attempts_on_question % len(paraphrases)

        # ====== Difficulty Visualization ======
        # fig = px.histogram(result_df, x="difficulty_score", nbins=20, title="Difficulty Distribution")
        # fig.update_layout(xaxis_title="Difficulty Score (1-10)", yaxis_title="Number of Questions")
        # st.plotly_chart(fig, use_container_width=True)
        
        # Timer
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()
        elapsed_time = time.time() - st.session_state.start_time
        st.markdown(f"⏱ **Time Elapsed**: {int(elapsed_time)} seconds")

        st.markdown(f"### ❓ Question {current_index + 1} of {len(result_df)}")
        st.write(paraphrases[phrasing_index])

        choices = {
            "A": row['choiceA'],
            "B": row['choiceB'],
            "C": row['choiceC'],
            "D": row['choiceD'],
            "E": row['choiceE'],
        }

        user_answer = st.radio(
            "Choose your answer:",
            options=list(choices.keys()),
            format_func=lambda x: f"{x}) {choices[x]}",
            key=f"question_{current_index}"
        )

        if st.button("Submit Answer"):
            correct_answer = row['answerKey'].strip().upper()
            st.session_state.time_taken.append(elapsed_time)
            st.session_state.start_time = None  # Reset timer
            if user_answer == correct_answer:
                st.session_state.answered_correctly.append(current_index)
                st.session_state.current_question += 1
                st.session_state.attempts_on_question = 0
                st.success("✅ Correct! Moving to next question...")
                st.rerun()
            else:
                st.session_state.attempts_on_question += 1
                # st.error(f"❌ Incorrect. Try again with a new phrasing!")
                st.rerun()

        # ====== Progress ======
        st.progress(st.session_state.current_question / len(result_df))
        st.markdown(f"**Progress**: {st.session_state.current_question}/{len(result_df)} questions answered")
    

        if st.session_state.show_correct:
            st.success("✅ Correct!")
            if time.time() - st.session_state.last_answer_time >= 2:
                st.session_state.answered_correctly.append(current_index)
                st.session_state.current_question += 1
                st.session_state.show_correct = False
                st.rerun()

    else:
        st.balloons()
        st.success("🎉 You've completed all questions!")
        
        # ====== Summary Statistics ======
        st.subheader("📊 Your Performance Summary")
        correct_count = len(st.session_state.answered_correctly)
        total_questions = len(result_df)
        accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        avg_time = np.mean(st.session_state.time_taken) if st.session_state.time_taken else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.2f}%")
        col2.metric("Questions Answered", f"{correct_count}/{total_questions}")
        col3.metric("Average Time per Question", f"{avg_time:.2f} seconds")

        if st.button("🔁 Restart"):
            st.session_state.current_question = 0
            st.session_state.answered_correctly = []
            st.session_state.attempts_on_question = 0
            st.session_state.time_taken = []
            st.session_state.start_time = None
            st.rerun()

    # ====== Download Results ======
    download_df = result_df[["question", "choiceA", "choiceB", "choiceC", "choiceD", "choiceE", "answerKey"]]
    csv = download_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download sorted questions as CSV",
        data=csv,
        file_name="sorted_questions.csv",
        mime="text/csv"
    )

# ====== Manual Input Feature ======
st.markdown("---")
st.subheader("📝 Manual Question Input")
with st.form(key="manual_input_form"):
    question = st.text_input("Enter your question:")
    choice_a = st.text_input("Choice A:")
    choice_b = st.text_input("Choice B:")
    choice_c = st.text_input("Choice C:")
    choice_d = st.text_input("Choice D:")
    choice_e = st.text_input("Choice E:")
    answer_key = st.selectbox("Correct Answer:", ["A", "B", "C", "D", "E"])
    submit_manual = st.form_submit_button("Add Question")

    if submit_manual:
        if all([question, choice_a, choice_b, choice_c, choice_d, choice_e, answer_key]):
            manual_df = pd.DataFrame([{
                "question": question,
                "choiceA": choice_a,
                "choiceB": choice_b,
                "choiceC": choice_c,
                "choiceD": choice_d,
                "choiceE": choice_e,
                "answerKey": answer_key
            }])
            if "result_df" in locals():
                result_df = pd.concat([result_df, compute_difficulty(manual_df)], ignore_index=True)
            else:
                result_df = compute_difficulty(manual_df)
            st.success("✅ Question added successfully!")
        else:
            st.error("❌ Please fill in all fields.")