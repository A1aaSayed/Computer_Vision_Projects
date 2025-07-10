import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import time
import plotly.express as px
from datetime import datetime
import logging
try:
    from pylatex import Document, Section, Command, Package, Tabular, Itemize
except ImportError:
    Document = Section = Command = Package = Tabular = Itemize = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(page_title="MCQ Difficulty Estimator", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for black text on light background, white text on dark sidebar
st.markdown("""
    <style>
    .main { 
        background-color: #f5f7fa; 
        color: #000000 !important; 
    }
    section[data-testid="stSidebar"] { 
        background-color: #2c3e50 !important; 
    }
    section[data-testid="stSidebar"] *, 
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] div { 
        color: white !important; 
    }
    .stButton>button { 
        background-color: #007bff; 
        color: white !important; 
        border-radius: 5px; 
    }
    .stButton>button:hover { 
        background-color: #0056b3; 
        color: white !important; 
    }
    .stRadio, .stRadio label, .stRadio div, .stRadio input { 
        font-size: 16px; 
        color: #000000 !important; 
    }
    .stSelectbox, .stSelectbox label, .stSelectbox div { 
        color: #000000 !important; 
    }
    .stCheckbox, .stCheckbox label, .stCheckbox div { 
        color: #000000 !important; 
    }
    .question-container { 
        background-color: #ffffff;
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
        color: #000000 !important; 
    }
    .metric-container { 
        background-color: #e9ecef; 
        padding: 10px; 
        border-radius: 5px; 
        color: #000000 !important; 
    }
    h1, h2, h3, h4, h5, h6 { 
        color: #000000 !important; 
        font-family: 'Arial', sans-serif; 
    }
    .stProgress .st-bo { 
        background-color: #007bff; 
    }
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span { 
        color: #000000 !important;
    }
    .stSuccess, .stError, .stWarning, .stInfo, 
    .stSuccess div, .stError div, .stWarning div, .stInfo div { 
        color: #000000 !important;
    }
    /* Added for metrics */
    [data-testid="stMetric"], 
    [data-testid="stMetricLabel"], 
    [data-testid="stMetricValue"], 
    [data-testid="stMetricDelta"] { 
        color: #000000 !important; 
    }
    /* Download button */
    [data-testid="stDownloadButton"] button { 
        background-color: #b3d7ff !important; 
        color: #000000 !important; 
        border-radius: 5px; 
        border: 1px solid #ffffff !important;
        padding: 8px 16px; 
    }
    [data-testid="stDownloadButton"] button:hover { 
        background-color: #003c80 !important; 
        color: #ffffff !important; 
        border: 1px solid #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize BERT
@st.cache_resource
def load_bert():
    """Load fine-tuned BERT tokenizer and model from local folder."""
    try:
        tokenizer = BertTokenizer.from_pretrained("bert_model")
        model = BertForSequenceClassification.from_pretrained("bert_model")
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading fine-tuned BERT: {e}")
        st.markdown(
            """
            <div style="background-color: #f8d7da; padding: 10px; border-left: 6px solid #dc3545; border-radius: 5px; color: #000000;">
                <span style="color: #000000;">❌ Failed to load fine-tuned BERT model from 'bert_model' folder. Ensure the folder exists and contains model files.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        return None, None

tokenizer, model = load_bert()
if tokenizer is None or model is None:
    st.stop()

# Prediction
label_mapping = {
    0: "Easy",
    1: "Hard",
    2: "Medium",
    3: "Very_Hard"
}
difficulty_scores = {
    "Easy": 2.0,
    "Medium": 4.0,
    "Hard": 6.0,
    "Very_Hard": 8.0
}

def predict_difficulty(text):
    """Predict difficulty level using fine-tuned BERT model."""
    try:
        if not isinstance(text, str) or pd.isna(text) or text.strip() == "":
            return "Easy"  # Default to Easy for invalid inputs
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        return label_mapping[predicted_class]
    except Exception as e:
        logger.error(f"Error predicting difficulty: {e}")
        return "Easy"  # Fallback to Easy

# Generate paraphrases
def generate_paraphrases(question):
    """Generate paraphrases for a given question."""
    if not isinstance(question, str) or pd.isna(question) or question.strip() == "":
        return [""]
    return [
        question,
        f"Explain this: {question}",
        f"Solve this: {question}",
        f"Answer the following: {question}",
        f"Consider this question: {question}"
    ]

# Compute difficulty
def compute_difficulty(df):
    """Compute difficulty levels and scores using BERT classifier."""
    try:
        df["full_text"] = df.apply(
            lambda row: f"{row['question']} A) {row['choiceA']} B) {row['choiceB']} C) {row['choiceC']} D) {row['choiceD']}",
            axis=1
        )
        df["difficulty_level"] = df["full_text"].apply(predict_difficulty)
        df["difficulty_score"] = df["difficulty_level"].map(difficulty_scores)
        df["paraphrases"] = df["question"].apply(generate_paraphrases)
        difficulty_order = {"Easy": 0, "Medium": 1, "Hard": 2, "Very_Hard": 3}
        df["difficulty_order"] = df["difficulty_level"].map(difficulty_order)
        df["difficulty"] = df["difficulty_level"]
        df = df.sort_values("difficulty_order").reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Error computing difficulty: {e}")
        st.markdown(
            """
            <div style="background-color: #f8d7da; padding: 10px; border-left: 6px solid #dc3545; border-radius: 5px; color: #000000;">
                <span style="color: #000000;">❌ Error processing questions. Please check your dataset.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        return df

# Generate LaTeX performance report
def generate_performance_report(correct_count, total_questions, accuracy, avg_time, difficulty_counts):
    """Generate a LaTeX performance report."""
    try:
        latex_content = r"""
        \documentclass[a4paper,12pt]{article}
        \usepackage{geometry}
        \usepackage{amsmath}
        \usepackage{booktabs}
        \usepackage{times}
        \geometry{margin=1in}
        \begin{document}
        \begin{center}
        \textbf{\Large MCQ Difficulty Estimator Performance Report} \\
        \vspace{0.5cm}
        Generated on \today \\
        \end{center}
        
        \section*{Summary}
        \begin{itemize}
            \item \textbf{Total Questions Answered}: %d
            \item \textbf{Correct Answers}: %d
            \item \textbf{Accuracy}: %.2f\%
            \item \textbf{Average Time per Question}: %.2f seconds
        \end{itemize}
        
        \section*{Difficulty Breakdown}
        \begin{table}[h]
        \centering
        \begin{tabular}{l c}
        \toprule
        \textbf{Difficulty} & \textbf{Count} \\
        \midrule
        Easy & %d \\
        Medium & %d \\
        Hard & %d \\
        Very Hard & %d \\
        \bottomrule
        \end{tabular}
        \caption{Number of questions per difficulty level}
        \end{table}
        
        \section*{Notes}
        This report summarizes your performance on the MCQ Difficulty Estimator. Continue practicing to improve your skills!
        
        \end{document}
        """ % (total_questions, correct_count, accuracy, avg_time, 
               difficulty_counts.get('Easy', 0), 
               difficulty_counts.get('Medium', 0), 
               difficulty_counts.get('Hard', 0), 
               difficulty_counts.get('Very_Hard', 0))
        
        if Document is None:
            return latex_content, None
        
        doc = Document()
        doc.packages.append(Package('geometry', options=['margin=1in']))
        doc.packages.append(Package('amsmath'))
        doc.packages.append(Package('booktabs'))
        doc.packages.append(Package('times'))
        
        with doc.create(Section('Summary', numbering=False)):
            with doc.create(Itemize()) as itemize:
                itemize.add_item(f"Total Questions Answered: {total_questions}")
                itemize.add_item(f"Correct Answers: {correct_count}")
                itemize.add_item(f"Accuracy: {accuracy:.2f}\\%")
                itemize.add_item(f"Average Time per Question: {avg_time:.2f} seconds")
        
        with doc.create(Section('Difficulty Breakdown', numbering=False)):
            with doc.create(Tabular('l c')) as table:
                table.add_hline()
                table.add_row(('Difficulty', 'Count'))
                table.add_hline()
                table.add_row(('Easy', difficulty_counts.get('Easy', 0)))
                table.add_row(('Medium', difficulty_counts.get('Medium', 0)))
                table.add_row(('Hard', difficulty_counts.get('Hard', 0)))
                table.add_row(('Very Hard', difficulty_counts.get('Very_Hard', 0)))
                table.add_hline()
        
        with doc.create(Section('Notes', numbering=False)):
            doc.append('This report summarizes your performance on the MCQ Difficulty Estimator. Continue practicing to improve your skills!')
        
        pdf = doc.dumps_as_content()
        return latex_content, pdf
    except Exception as e:
        logger.error(f"Error generating LaTeX report: {e}")
        st.markdown(
            """
            <div style="background-color: #fff3cd; padding: 10px; border-left: 6px solid #ffc107; border-radius: 5px; color: #000000;">
                <span style="color: #000000;">⚠️ Error generating LaTeX report: %s. LaTeX source will be provided.</span>
            </div>
            """ % str(e),
            unsafe_allow_html=True
        )
        return latex_content, None

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        "current_question": 0,
        "answered_correctly": [],
        "attempts_on_question": 0,
        "start_time": None,
        "show_correct": False,
        "last_answer_time": None,
        "time_taken": [],
        "difficulty_filter": "All",
        "adaptive_difficulty": False,
        "score": 0,
        "file_uploader_key": 0  # Track file uploader state
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Sidebar for settings
with st.sidebar:
    st.markdown('<h1 style="color: white !important;">⚙️ Settings</h1>', unsafe_allow_html=True)
    st.markdown('<div style="color: white !important;">Filter by Difficulty:</div>', unsafe_allow_html=True)
    st.session_state.difficulty_filter = st.selectbox(
        "",
        ["All", "Easy", "Medium", "Hard", "Very_Hard"],
        index=["All", "Easy", "Medium", "Hard", "Very_Hard"].index(st.session_state.difficulty_filter),
        key="difficulty_select"
    )
    st.session_state.adaptive_difficulty = st.checkbox(
        "Enable Adaptive Difficulty",
        value=st.session_state.adaptive_difficulty
    )
    if st.button("Reset Session"):
        # Clear all session state and increment file uploader key
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_session_state()
        st.session_state.file_uploader_key += 1
        logger.info("Session reset triggered")
        st.rerun()

# Main UI
st.markdown('<h1 style="color: #000000;">📚 MCQ Difficulty Estimator</h1>', unsafe_allow_html=True)
st.markdown('<div style="color: #000000;">Enhance your knowledge with adaptive questions and real-time feedback!</div>', unsafe_allow_html=True)

# File uploader
st.markdown('<div style="color: #000000;">📂 Upload MCQ File (CSV or Excel)</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv", "xlsx"], key=f"file_uploader_{st.session_state.file_uploader_key}")

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        df.columns = df.columns.str.strip()
        logger.info(f"Loaded file with columns: {df.columns.tolist()}")
        
        required_cols = {"question", "choiceA", "choiceB", "choiceC", "choiceD", "answerKey"}
        if not required_cols.issubset(set(df.columns)):
            st.markdown(
                """
                <div style="background-color: #f8d7da; padding: 10px; border-left: 6px solid #dc3545; border-radius: 5px; color: #000000;">
                    <span style="color: #000000;">❌ File must include columns: question, choiceA, choiceB, choiceC, choiceD, answerKey.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.stop()
        
        # Handle difficulty column
        if 'difficulty' not in df.columns:
            df['difficulty'] = 'unknown'
        
        # Clean data
        df = df.dropna(subset=["question", "answerKey"])
        for col in ["question", "choiceA", "choiceB", "choiceC", "choiceD"]:
            df[col] = df[col].astype(str).replace("nan", "")
        
        st.markdown(
            """
            <div style="background-color: #d4edda; padding: 10px; border-left: 6px solid #28a745; border-radius: 5px; color: #000000;">
                <span style="color: #000000;">✅ File loaded successfully. Calculating difficulty scores...</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        result_df = compute_difficulty(df)
        
        # Filter by difficulty
        if st.session_state.difficulty_filter != "All":
            result_df = result_df[result_df['difficulty_level'] == st.session_state.difficulty_filter]
        if result_df.empty:
            st.markdown(
                """
                <div style="background-color: #fff3cd; padding: 10px; margin-top:15px; border-left: 6px solid #ffc107; border-radius: 5px; color: #000000;">
                    <span style="color: #000000;">⚠️ No questions match the selected difficulty. Showing all questions.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            result_df = compute_difficulty(df)
        
        # Adaptive difficulty adjustment
        if st.session_state.adaptive_difficulty and len(st.session_state.answered_correctly) >= 3:
            recent_correct = st.session_state.answered_correctly[-3:]
            recent_accuracy = len([i for i in recent_correct if i in st.session_state.answered_correctly]) / 3
            if recent_accuracy > 0.7:
                result_df = result_df[result_df['difficulty_order'] >= result_df['difficulty_order'].quantile(0.7)]
            elif recent_accuracy < 0.3:
                result_df = result_df[result_df['difficulty_order'] <= result_df['difficulty_order'].quantile(0.3)]
        
        # Display current question
        current_index = st.session_state.current_question
        if current_index < len(result_df):
            row = result_df.iloc[current_index]
            paraphrases = row["paraphrases"]
            phrasing_index = st.session_state.attempts_on_question % len(paraphrases)
            
            # Timer
            if st.session_state.start_time is None:
                st.session_state.start_time = time.time()
            elapsed_time = time.time() - st.session_state.start_time
            st.markdown(f'<div style="color: #000000;">⏱ <b>Time Elapsed</b>: {int(elapsed_time)} seconds</div>', unsafe_allow_html=True)
            
            # Question display
            with st.container():
                st.markdown(f'<h3 style="color: #000000;">❓ Question {current_index + 1} of {len(result_df)} (Difficulty: {row["difficulty"]}, Score: {row["difficulty_score"]:.1f})</h3>', unsafe_allow_html=True)
                st.markdown(f'<div style="color: #000000;">{paraphrases[phrasing_index]}</div>', unsafe_allow_html=True)
                
                choices = {
                    "A": row['choiceA'],
                    "B": row['choiceB'],
                    "C": row['choiceC'],
                    "D": row['choiceD'],
                }
                
                st.markdown('<div style="color: #000000;">Choose your answer:</div>', unsafe_allow_html=True)
                user_answer = st.radio(
                    "",
                    options=list(choices.keys()),
                    format_func=lambda x: f"{x}) {choices[x]}",
                    key=f"question_{current_index}_{phrasing_index}"
                )
                
                if st.button("Submit Answer", key=f"submit_{current_index}"):
                    correct_answer = row['answerKey'].strip().upper()
                    st.session_state.time_taken.append(elapsed_time)
                    st.session_state.start_time = None
                    if user_answer == correct_answer:
                        st.session_state.answered_correctly.append(current_index)
                        st.session_state.score += row['difficulty_score']
                        st.session_state.current_question += 1
                        st.session_state.attempts_on_question = 0
                        st.markdown(
                            """
                            <div style="background-color: #d4edda; padding: 10px; border-left: 6px solid #28a745; border-radius: 5px; color: #000000;">
                                <span style="color: #000000;">✅ Correct! Moving to next question...</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.session_state.attempts_on_question += 1
                        st.markdown(
                            """
                            <div style="background-color: #f8d7da; padding: 10px; border-left: 6px solid #dc3545; border-radius: 5px; color: #000000;">
                                <span style="color: #000000;">❌ Incorrect. Try again with a new phrasing!</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        time.sleep(1)
                        st.rerun()
                
                # Progress and metrics
                st.progress(min(st.session_state.current_question / len(result_df), 1.0))
                col1, col2, col3 = st.columns(3)
                col1.metric("Progress", f"{st.session_state.current_question}/{len(result_df)}")
                col2.metric("Score", f"{st.session_state.score:.1f}")
                col3.metric("Correct Answers", len(st.session_state.answered_correctly))
                
                # Difficulty distribution plot
                with st.expander("📊 View Difficulty Distribution"):
                    fig = px.histogram(result_df, x="difficulty_level", title="Question Difficulty Distribution")
                    fig.update_layout(xaxis_title="Difficulty Level", yaxis_title="Number of Questions")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance over time
                with st.expander("📈 Performance Over Time"):
                    if st.session_state.time_taken:
                        performance_data = pd.DataFrame({
                            "Question": range(1, len(st.session_state.time_taken) + 1),
                            "Time Taken (s)": st.session_state.time_taken,
                            "Correct": [i in st.session_state.answered_correctly for i in range(len(st.session_state.time_taken))]
                        })
                        fig = px.scatter(performance_data, x="Question", y="Time Taken (s)", color="Correct",
                                        title="Time Taken per Question")
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.balloons()
            st.markdown(
                """
                <div style="background-color: #d4edda; padding: 10px; margin-top:15px; border-left: 6px solid #28a745; border-radius: 5px; color: #000000;">
                    <span style="color: #000000;">🎉 You've completed all questions!</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Summary statistics
            st.markdown('<h2 style="color: #000000;">📊 Performance Summary</h2>', unsafe_allow_html=True)
            correct_count = len(st.session_state.answered_correctly)
            total_questions = len(result_df)
            accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
            avg_time = np.mean(st.session_state.time_taken) if st.session_state.time_taken else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.2f}%", delta=f"{accuracy - 50:.2f}%", delta_color="normal")
            col2.metric("Questions Answered", f"{correct_count}/{total_questions}")
            col3.metric("Average Time", f"{avg_time:.2f} seconds")
            col4.metric("Total Score", f"{st.session_state.score:.1f}")
            
            
            # Download sorted questions
            try:
                if all(col in result_df.columns for col in ["question", "choiceA", "choiceB", "choiceC", "choiceD", "answerKey"]):
                    download_df = result_df[["question", "choiceA", "choiceB", "choiceC", "choiceD", "answerKey"]]
                    csv = download_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Sorted Questions (CSV)",
                        data=csv,
                        file_name="sorted_questions.csv",
                        mime="text/csv"
                    )
                else:
                    st.markdown(
                        """
                        <div style="background-color: #f8d7da; padding: 10px; border-left: 6px solid #dc3545; border-radius: 5px; color: #000000;">
                            <span style="color: #000000;">❌ Error: Missing required columns for CSV export.</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                logger.error(f"Error generating CSV: {e}")
                st.markdown(
                    """
                    <div style="background-color: #f8d7da; padding: 10px; border-left: 6px solid #dc3545; border-radius: 5px; color: #000000;">
                        <span style="color: #000000;">❌ Error generating CSV: %s. Please check the dataset.</span>
                    </div>
                    """ % str(e),
                    unsafe_allow_html=True
                )
            
            # if st.button("🔁 Restart"):
            #     preserved_keys = ['result_df', 'file_uploader_key']
            #     for key in list(st.session_state.keys()):
            #         if key not in preserved_keys:
            #             del st.session_state[key]
            #     initialize_session_state()
            #     logger.info("Quiz restart triggered, preserving uploaded file")
            #     st.rerun()
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.markdown(
            """
            <div style="background-color: #f8d7da; padding: 10px; border-left: 6px solid #dc3545; border-radius: 5px; color: #000000;">
                <span style="color: #000000;">❌ Error processing file: %s. Please ensure it is a valid CSV or Excel file.</span>
            </div>
            """ % str(e),
            unsafe_allow_html=True
        )
else:
    st.markdown(
        """
        <div style="background-color: #e1f5fe; padding: 10px; border-left: 6px solid #2196f3; border-radius: 5px; color: #000000;">
            <span style="color: #000000;">ℹ️ Upload a CSV or Excel file to start. Ensure it has columns: question, choiceA, choiceB, choiceC, choiceD, answerKey.</span>
        </div>
        """,
        unsafe_allow_html=True
    )