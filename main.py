# import faiss
# import json
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from gpt4all import GPT4All

# # Load FAISS index and document metadata
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# faiss_index = faiss.read_index('faiss_index.bin')

# with open("document_metadata.json", "r") as f:
#     documents = json.load(f)

# # Initialize GPT4All model
# gpt4all_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

# # Function to retrieve relevant documents
# def retrieve_relevant_docs(query, top_k=2):
#     query_embedding = embed_model.encode([query])
#     _, indices = faiss_index.search(query_embedding, top_k)
#     relevant_docs = [documents[i] for i in indices[0]]
#     return relevant_docs

# # Generate answer in chunks
# def generate_answer(query):
#     relevant_docs = retrieve_relevant_docs(query)
#     context = "\n\n".join([doc["content"] for doc in relevant_docs])

#     # Trim context if it's too long for a single prompt
#     max_context_words = 300
#     if len(context.split()) > max_context_words:
#         context = " ".join(context.split()[:max_context_words])

#     # Use prompt length management
#     prompt = f"Answer the question based on the context provided.\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
#     response_chunks = []
#     try:
#         response = gpt4all_model.generate(prompt, max_tokens=300)  # Adjust max_tokens to limit response size
#         response_chunks.append(response)
        
#         # Check if the response is still cut off and fetch additional chunks if needed
#         while "..." in response[-3:]:  # Detect cut-off response
#             prompt = "Continue the answer based on the previous context."
#             response = gpt4all_model.generate(prompt, max_tokens=300)
#             response_chunks.append(response)
#             if "..." not in response[-3:]:
#                 break
#     except Exception as e:
#         st.error("Error generating response from model.")
    
#     return " ".join(response_chunks), relevant_docs

# # Set up Streamlit page configuration
# st.set_page_config(
#     page_title="Chat with Local LLM",
#     layout="centered",
#     initial_sidebar_state="auto",
# )

# st.title("Chat with Local LLM for Network Security")

# # Initialize session state for chat messages
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {
#             "role": "assistant",
#             "content": "I'm here to answer your questions about network security!"
#         }
#     ]

# # Get user input
# if prompt := st.chat_input("Your question:"):
#     st.session_state.messages.append({
#         "role": "user",
#         "content": prompt
#     })

#     # Generate the response
#     with st.spinner("Thinking..."):
#         answer, docs = generate_answer(prompt)
#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": answer
#         })

# # Display messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         response_content = message["content"]

#         # Display in chunks if the response is too long
#         if len(response_content) > 2000:
#             response_chunks = [response_content[i:i+2000] for i in range(0, len(response_content), 2000)]
#             for chunk in response_chunks:
#                 st.write(chunk)
#         else:
#             st.write(response_content)

# docs = []
# # Display citations if any
# if docs:
#     st.write("### Citations")
#     for doc in docs:
#         st.write(f"- {doc['filename']}")

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json
from pathlib import Path
import re
import fitz  # PyMuPDF

# Initialize the embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Load documents and create embeddings
doc_dir = "./data"  # Ensure that you place all course materials here
documents = []
for file_path in Path(doc_dir).glob("*.pdf"):
    content = load_and_clean_data(file_path)
    documents.append({
        "content": content,
        "filename": file_path.name
    })

# Initialize FAISS index
dimension = embed_model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(dimension)

# Embed and index documents
for doc in documents:
    embeddings = embed_model.encode([doc["content"]])
    faiss_index.add(embeddings)
    doc["embedding"] = embeddings.tolist()  # Convert ndarray to list for JSON serialization

# Save FAISS index and document metadata
faiss.write_index(faiss_index, 'faiss_index.bin')
with open("document_metadata.json", "w") as f:
    json.dump(documents, f)

# Initialize session state for storing messages if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit interface for user prompt input
if prompt := st.chat_input("Your question:"):
    # Step 1: Capture the user prompt
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    print("Captured Prompt:", prompt)

    # Step 2: Create embedding for the user prompt
    query_embedding = embed_model.encode([prompt])
    print("Query Embedding:", query_embedding)

    # Step 3: Search the FAISS index for relevant documents
    _, indices = faiss_index.search(query_embedding, k=2)  # Use 'k' instead of 'top_k'
    relevant_docs = [documents[idx] for idx in indices[0]]
    print("Retrieved Document Indices:", indices)
    print("Relevant Documents:", [doc["filename"] for doc in relevant_docs])

    # Step 4: Create the LLM prompt by combining context and the user query
    context = "\n\n".join([doc["content"] for doc in relevant_docs])
    prompt_for_llm = f"Answer the question based on the context provided.\nContext:\n{context}\n\nQuestion: {prompt}\nAnswer:"
    print("Prompt for LLM:", prompt_for_llm)

    # Simulating LLM response generation (replace with actual LLM call)
    response = "This is a simulated LLM response based on the context." + response

    # Step 5: Capture the LLM response
    print("LLM Response:", response)

    # Trace data to map Step 1 to Step 4
    trace_data = {
        "user_prompt": prompt,
        "query_embedding": query_embedding.tolist(),
        "retrieved_docs": [doc["filename"] for doc in relevant_docs],
        "llm_prompt": prompt_for_llm,
        "llm_response": response
    }

    # Save trace data to a log file
    with open("trace_log.json", "w") as f:
        json.dump(trace_data, f, indent=4)

    # Display the LLM response in Streamlit UI
    st.write(response)
    print("Trace Data Captured:", trace_data)
