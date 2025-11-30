import streamlit as st
import time
from rag_engine import query_rag_system

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Auto-Grader | Exam Mode",
    page_icon="📝",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
# This keeps the questions in memory even when the app refreshes
if "exam_questions" not in st.session_state:
    st.session_state.exam_questions = []

# --- SIDEBAR: TEACHER CONTROL PANEL ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/8b/Amrita_Vishwa_Vidyapeetham_Logo.svg/1200px-Amrita_Vishwa_Vidyapeetham_Logo.svg.png", width=100)
    st.title("👨‍🏫 Teacher Panel")
    st.markdown("**1. Set Exam Questions**")
    
    # Input area for teacher to paste questions
    default_questions = "What is Zero-Shot Prompting?\nExplain Chain-of-Thought prompting.\nWhat is QLoRA?"
    teacher_input = st.text_area("Enter questions (one per line):", value=default_questions, height=150)
    
    if st.button("🚀 Publish Test"):
        # Split by new lines and remove empty lines
        questions = [q.strip() for q in teacher_input.split('\n') if q.strip()]
        st.session_state.exam_questions = questions
        st.success(f"Test Published with {len(questions)} questions!")

    st.divider()
    st.info("The student will see these questions on the main screen.")

# --- MAIN PAGE: STUDENT EXAM INTERFACE ---
st.title("📝 Semester Exam: Prompt Engineering")
st.markdown("Please answer the following questions. The AI will grade your responses against the **Course Textbook**.")

# Check if questions exist
if not st.session_state.exam_questions:
    st.warning("⚠️ Waiting for the Professor to publish the test questions...")
else:
    # --- FORM FOR STUDENT ANSWERS ---
    with st.form("exam_form"):
        student_answers = {}
        
        # Loop through questions and create input boxes
        for i, question in enumerate(st.session_state.exam_questions):
            st.markdown(f"**Q{i+1}: {question}**")
            # We use a unique key for each text area so Streamlit tracks them separately
            student_answers[question] = st.text_area(f"Your Answer for Q{i+1}:", key=f"q_{i}")
            st.markdown("---")
        
        # Submit Button
        submitted = st.form_submit_button("✅ Submit Test for Grading")

    # --- GRADING LOGIC ---
    if submitted:
        st.subheader("📊 Your Results")
        progress_bar = st.progress(0)
        total_q = len(st.session_state.exam_questions)
        
        for idx, (question, answer) in enumerate(student_answers.items()):
            # 1. Format the input for the RAG Brain
            # We combine Q and A so the AI has full context
            rag_input = f"Question: {question} Answer: {answer}"
            
            # 2. Get AI Response
            with st.spinner(f"Grading Question {idx+1}..."):
                try:
                    result = query_rag_system(rag_input)
                except Exception as e:
                    result = f"Error: {e}"
            
            # 3. Display Result nicely
            with st.expander(f"Q{idx+1}: {question}", expanded=True):
                st.markdown(f"**Your Answer:** {answer}")
                st.markdown("### 👨‍🏫 AI Feedback:")
                st.markdown(result)
            
            # Update progress bar
            progress_bar.progress((idx + 1) / total_q)
        
        st.success("🎉 Grading Complete!")