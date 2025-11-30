import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
DB_PATH = os.path.join("..", "data", "chroma_db")
COLLECTION_NAME = "university_rules" # We keep the same name for simplicity
MODEL_NAME = "llama3" 

def get_retriever():
    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    return collection

def query_rag_system(user_query):
    # 1. RETRIEVAL
    collection = get_retriever()
    results = collection.query(
        query_texts=[user_query],
        n_results=3
    )
    
    context_text = "\n\n".join(results['documents'][0])
    
    # 2. PROMPTING: The "Strict Professor" Persona
    prompt_template = ChatPromptTemplate.from_template("""
    You are a strict Computer Science Professor grading a student's answer about LLMs and Prompt Engineering.
    
    ### REFERENCE MATERIAL (TEXTBOOK CONTENT):
    {context}
    
    ### STUDENT Q&A SUBMISSION:
    {question}
    
    ### GRADING INSTRUCTIONS:
    1. Analyze the Student's Answer based ONLY on the Reference Material.
    2. Check for:
       - **Keywords:** Did they use technical terms (e.g., Zero-shot, Chain-of-Thought, QLoRA)?
       - **Accuracy:** Is the definition factually correct according to the text?
       - **Completeness:** Did they miss any key points?
    3. Ignore spelling mistakes, focus on the Logic.
    
    ### OUTPUT FORMAT (Markdown):
    **Grade:** [Score out of 10]
    
    **Analysis:**
    [Provide 2 sentences explaining what is correct]
    
    **Missing/Incorrect:**
    [Point out exactly what concepts were missing from the textbook]
    
    **Ideal Answer:**
    [Write the perfect 1-sentence answer based on the Reference]
    """)

    # 3. GENERATION
    model = ChatOllama(model=MODEL_NAME)
    chain = prompt_template | model | StrOutputParser()
    
    response = chain.invoke({
        "context": context_text,
        "question": user_query
    })
    
    return response

if __name__ == "__main__":
    # Test Query
    print("Testing Auto-Grader...")
    test_q = "Question: What is Zero-Shot Prompting? Answer: It is when you ask the AI to do something without giving examples."
    answer = query_rag_system(test_q)
    print(answer)