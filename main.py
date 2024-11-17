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


# 2ND ATTEMPT
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# import faiss
# import json
# from pathlib import Path
# import re
# import fitz  # PyMuPDF
# from gpt4all import GPT4All

# # Initialize the embedding and LLM models
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# gpt4all_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

# # Load FAISS index and document metadata
# faiss_index = faiss.read_index('faiss_index.bin')
# with open("document_metadata.json", "r") as f:
#     documents = json.load(f)

# # Function to retrieve relevant documents
# def retrieve_relevant_docs(query, top_k=2):
#     query_embedding = embed_model.encode([query])
#     _, indices = faiss_index.search(query_embedding, top_k)
#     relevant_docs = [documents[i] for i in indices[0]]
#     return relevant_docs

# # Generate answer in chunks from LLM
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

# #3RD ATTEMP
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# import faiss
# import json
# from pathlib import Path
# import re
# import fitz  # PyMuPDF
# from gpt4all import GPT4All

# # Initialize the embedding and LLM models
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# gpt4all_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

# # Load FAISS index and document metadata
# faiss_index = faiss.read_index('faiss_index.bin')
# with open("document_metadata.json", "r") as f:
#     documents = json.load(f)

# # Initialize log file
# log_file_path = Path("trace_log.txt")
# if not log_file_path.exists():
#     log_file_path.touch()

# # Function to log prompt, response, and document metadata
# def log_trace(prompt, response, docs):
#     with open(log_file_path, "a") as log_file:
#         log_file.write(f"Prompt: {prompt}\n")
#         log_file.write(f"Response: {response}\n")
#         if docs:
#             log_file.write("Relevant Documents:\n")
#             for doc in docs:
#                 log_file.write(f" - {doc['filename']}: {doc['content'][:200]}...\n")  # Only log a snippet of each document for brevity
#         log_file.write("\n" + "-" * 50 + "\n\n")

# # Function to retrieve relevant documents
# def retrieve_relevant_docs(query, top_k=2):
#     query_embedding = embed_model.encode([query])
#     _, indices = faiss_index.search(query_embedding, top_k)
#     relevant_docs = [documents[i] for i in indices[0]]
#     return relevant_docs

# # Generate answer in chunks from LLM
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
    
#     # Combine response and log it
#     final_response = " ".join(response_chunks)
#     log_trace(query, final_response, relevant_docs)
#     return final_response, relevant_docs

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


#4th attempt
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json
from pathlib import Path
import re
import fitz  # PyMuPDF
from gpt4all import GPT4All

# Initialize the embedding and LLM models
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
gpt4all_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

# Load FAISS index and document metadata
faiss_index = faiss.read_index('faiss_index.bin')
with open("document_metadata.json", "r") as f:
    documents = json.load(f)

# Initialize log file
log_file_path = Path("trace_log.txt")
if not log_file_path.exists():
    log_file_path.touch()

# Function to log prompt, response, and document metadata
def log_trace(trace_data):
    with open(log_file_path, "a") as log_file:
        json.dump(trace_data, log_file, indent=4)
        log_file.write("\n" + "-" * 50 + "\n\n")

# Function to retrieve relevant documents
def retrieve_relevant_docs(query, top_k=2):
    query_embedding = embed_model.encode([query])
    _, indices = faiss_index.search(query_embedding, top_k)
    relevant_docs = [documents[i] for i in indices[0]]
    return relevant_docs, query_embedding

# Generate answer in chunks from LLM
def generate_answer(query):
    relevant_docs, query_embedding = retrieve_relevant_docs(query)
    context = "\n\n".join([doc["content"] for doc in relevant_docs])

    # Trim context if it's too long for a single prompt
    max_context_words = 300
    if len(context.split()) > max_context_words:
        context = " ".join(context.split()[:max_context_words])

    # Prepare LLM prompt and response
    prompt_for_llm = f"Answer the question based on the context provided.\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response_chunks = []
    try:
        response = gpt4all_model.generate(prompt_for_llm, max_tokens=300)  # Adjust max_tokens to limit response size
        response_chunks.append(response)
        
        # Check if the response is still cut off and fetch additional chunks if needed
        while "..." in response[-3:]:  # Detect cut-off response
            prompt_for_llm = "Continue the answer based on the previous context."
            response = gpt4all_model.generate(prompt_for_llm, max_tokens=300)
            response_chunks.append(response)
            if "..." not in response[-3:]:
                break
    except Exception as e:
        st.error("Error generating response from model.")
    
    # Combine response and prepare trace data
    final_response = " ".join(response_chunks)
    trace_data = {
        "user_prompt": query,
        "query_embedding": query_embedding.tolist(),
        "retrieved_docs": [doc["filename"] for doc in relevant_docs],
        "llm_prompt": prompt_for_llm,
        "llm_response": final_response
    }
    log_trace(trace_data)  # Log the structured trace data
    return final_response, relevant_docs

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Chat with Local LLM",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("Chat with Local LLM for Network Security")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "I'm here to answer your questions about network security!"
        }
    ]

# Get user input
if prompt := st.chat_input("Your question:"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Generate the response
    with st.spinner("Thinking..."):
        answer, docs = generate_answer(prompt)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        response_content = message["content"]

        # Display in chunks if the response is too long
        if len(response_content) > 2000:
            response_chunks = [response_content[i:i+2000] for i in range(0, len(response_content), 2000)]
            for chunk in response_chunks:
                st.write(chunk)
        else:
            st.write(response_content)

docs = []
# Display citations if any
if docs:
    st.write("### Citations")
    for doc in docs:
        st.write(f"- {doc['filename']}")
