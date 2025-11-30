import json
import os
import chromadb
from chromadb.utils import embedding_functions

# --- CONFIGURATION ---
CHUNKS_PATH = os.path.join("..", "data", "chunks", "rules_chunked.json")
DB_PATH = os.path.join("..", "data", "chroma_db")
COLLECTION_NAME = "university_rules"

def load_chunks():
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"❌ Error: {CHUNKS_PATH} not found. Did you run preprocessing.py?")
    with open(CHUNKS_PATH, "r") as f:
        return json.load(f)

def build_database():
    print("--- 1. Initializing Vector Database ---")
    
    client = chromadb.PersistentClient(path=DB_PATH)
    
    print("   -> Loading AI Model (all-MiniLM-L6-v2)...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # --- FIXED SECTION ---
    # We try to delete the old collection. If it doesn't exist, we just move on.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"   -> Deleted existing collection '{COLLECTION_NAME}' to start fresh.")
    except Exception: 
        pass 
    # ---------------------

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )
    
    chunks = load_chunks()
    print(f"--- 2. Embedding {len(chunks)} chunks... ---")
    
    documents = [item['text'] for item in chunks]
    ids = [str(item['id']) for item in chunks]
    metadatas = [{"source": "Regulations 2023"} for _ in chunks]
    
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"✅ Success! Database built at {DB_PATH}")

def test_retrieval():
    print("\n--- 3. Testing Semantic Search ---")
    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    
    query = "What is the penalty for missing too many classes?"
    
    results = collection.query(
        query_texts=[query],
        n_results=1
    )
    
    print(f"🔎 Query: '{query}'")
    print(f"📝 Retrieved Rule: {results['documents'][0][0]}")

if __name__ == "__main__":
    build_database()
    test_retrieval()