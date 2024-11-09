# NS OnDemand Q&ABot

This is a local Q&A bot that leverages a local LLM model, FAISS, and embeddings to answer questions based on your network security course material. It’s built using Streamlit for a user-friendly UI.

Project Setup
Follow the steps below to set up and run this project on your local machine.

Prerequisites
Ensure you have the following installed:

Python 3.10 or higher
Git
Streamlit (for running the UI)
Installation
Clone the Repository

bash
Copy code
git clone https://github.com/Gop1Prudhv1/NS_OnDemand_QABot.git
cd NS_OnDemand_QABot
Set Up a Virtual Environment

It’s recommended to use a virtual environment to manage dependencies.

python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
Install Dependencies


pip install -r requirements.txt
Download Model Files

Ensure the GPT4All model, embeddings, and FAISS index files are correctly downloaded in your environment. If they are missing, you can generate them by following the instructions in ingestion.py.

Set Up FAISS Index and Document Metadata

Run the ingestion.py file to create faiss_index.bin and document_metadata.json if they’re not already present.

python ingestion.py
Note: These files are added to .gitignore since they are specific to the environment and can be re-generated.

Running the App - Launch Streamlit

streamlit run main.py


Access the App

After running the command, Streamlit will provide a URL. Open it in your browser to access the app. The URL should look something like:
arduino
Copy code
Local URL: http://localhost:8501

Using the App
Ask Questions: Enter your network security-related questions into the input box, and the model will respond with answers based on the uploaded course material.
Citations: The app provides citations for the documents it used to generate the answer.

Troubleshooting
FAISS Index Error: If you encounter a No such file or directory error for faiss_index.bin, make sure to run ingestion.py before running the app.
Dependencies Not Installing: Ensure your virtual environment is active before installing the requirements.
Additional Notes
The faiss_index.bin and document_metadata.json files are ignored in .gitignore to prevent them from being added to version control. Re-run ingestion.py if needed.
Ensure faiss_index.bin and document_metadata.json are in the project root directory when running the app.
