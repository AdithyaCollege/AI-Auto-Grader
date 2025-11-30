# 📝 AI Auto-Grader: Automated Subjective Answer Evaluation System

**An AI-powered grading assistant that evaluates student answers against reference textbooks using Retrieval-Augmented Generation (RAG).**

## 🚀 Overview
Manual grading of descriptive answers is time-consuming and subjective. This project automates the process using **Llama-3** (via Ollama) and **ChromaDB**. It acts as a "Strict Professor," comparing student submissions against a "Ground Truth" PDF (Textbook) to check for conceptual accuracy, missing keywords, and logical flow.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Python)
* **LLM Engine:** Llama-3 (via Ollama)
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB (Local)
* **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)

## ⚙️ How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/AdithyaCollege/AI-Auto-Grader.git](https://github.com/AdithyaCollege/AI-Auto-Grader.git)
    cd AI-Auto-Grader
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Data**
    * Place your course textbook PDF in `data/raw_pdfs/`.
    * Run the preprocessing script:
        ```bash
        cd src
        python preprocessing.py
        python build_vector_db.py
        ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## 📊 Features
* **Teacher Mode:** Publish exam questions dynamically.
* **Student Mode:** Submit answers for real-time evaluation.
* **RAG Pipeline:** Retrieves exact context from the textbook to justify the grade.
* **Strict Grading:** Penalizes vague answers and hallucinations.
