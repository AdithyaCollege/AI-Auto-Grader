import os
import re
import json
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
DATA_FOLDER = os.path.join("..", "data", "raw_pdfs")
OUTPUT_CHUNKS_PATH = os.path.join("..", "data", "chunks", "rules_chunked.json")

def get_pdf_path():
    """
    Automatically finds the PDF in the folder, no matter what it is named.
    """
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ Error: Folder not found at {DATA_FOLDER}")
        return None

    # List all files in the directory
    files = os.listdir(DATA_FOLDER)
    
    # Find the first file that ends with .pdf
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ Error: No PDF file found in {DATA_FOLDER}")
        print(f"   Files currently there: {files}")
        return None
        
    # Use the first PDF found
    return os.path.join(DATA_FOLDER, pdf_files[0])

def clean_text(text):
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\.{4,}', '', text)
    return text.strip()

def load_and_process_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
    print(f"   -> Found {len(reader.pages)} pages...")
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            cleaned_page_text = clean_text(page_text)
            full_text += cleaned_page_text + "\n"
    return full_text

def create_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def main():
    print("--- 1. Auto-Detecting PDF ---")
    pdf_path = get_pdf_path()
    
    if not pdf_path:
        return # Stop if no PDF

    clean_content = load_and_process_pdf(pdf_path)
    
    print(f"--- 2. Splitting Text (Length: {len(clean_content)} chars) ---")
    chunks = create_chunks(clean_content)
    print(f"✅ Created {len(chunks)} chunks.")
    
    data_to_save = [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]
    
    os.makedirs(os.path.dirname(OUTPUT_CHUNKS_PATH), exist_ok=True)
    with open(OUTPUT_CHUNKS_PATH, "w") as f:
        json.dump(data_to_save, f, indent=4)
        
    print(f"💾 Saved chunks to {OUTPUT_CHUNKS_PATH}")

if __name__ == "__main__":
    main()