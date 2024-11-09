from pathlib import Path
import re
import faiss
from sentence_transformers import SentenceTransformer
import json
import fitz  # PyMuPDF

# Load and clean text data from a PDF file
def load_and_clean_data(file_path):
    with fitz.open(file_path) as pdf_doc:
        content = ""
        for page in pdf_doc:
            content += page.get_text()

    # Basic text cleanup
    content = re.sub(r"\\n|\\u[\dA-Fa-f]{4}|—", "", content)
    content = re.sub(r"\s+", " ", content)
    return content

# Split text into chunks of approximately 500 words
def split_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Initialize embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and preprocess data
documents = []
doc_dir = "./data"  # Ensure that you place all course materials here
for file_path in Path(doc_dir).glob("*.pdf"):
    content = load_and_clean_data(file_path)
    chunks = split_into_chunks(content)  # Split content into 500-word chunks

    for i, chunk in enumerate(chunks):
        # Store each chunk as a separate document
        documents.append({
            "content": chunk,
            "filename": file_path.name,
            "chunk_id": i  # Track the chunk number within the document
        })

# Embed document chunks and index them with FAISS
dimension = embed_model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(dimension)

for doc in documents:
    embeddings = embed_model.encode([doc["content"]])
    faiss_index.add(embeddings)
    doc["embedding"] = embeddings.tolist()  # Convert ndarray to list for JSON serialization

# Save FAISS index and document metadata
faiss.write_index(faiss_index, 'faiss_index.bin')
with open("document_metadata.json", "w") as f:
    json.dump(documents, f)
