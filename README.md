# 📚 Advanced RAG (Retrieval-Augmented Generation) App

An advanced application leveraging NVIDIA's AI models for document embedding and querying. This app uses state-of-the-art language models to create an efficient, scalable retrieval system for your PDF documents.

## 🚀 Features

- **Document Embedding**: Convert PDF documents into vector embeddings for efficient information retrieval.
- **NVIDIA AI Integration**: Powered by NVIDIA's Llama-3.1-70b-instruct model for high-quality responses.
- **Dynamic Querying**: Query your embedded documents and retrieve contextually accurate answers.
- **Interactive UI**: Built with Streamlit, featuring a modern and user-friendly design.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: NVIDIA Llama-3.1, FAISS, LangChain
- **Environment**: Python, .env for API key management
- **Data Format**: PDF document handling via `PyPDFDirectoryLoader`

## 📁 Directory Structure

.
├── finalapp.py                  # Main application file
├── README.md               # Project documentation
├── requirements.txt        # Dependencies for the project
├── data/                   # Sample pdfs
├── RAG-Architecture.png    # Image for the architecture overview
└── .env                    # Environment variables (NVIDIA API Key)


## 📝 Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Project
2. **Install Dependencies: Create and activate a virtual environment, then install dependencies:

   ```bash
    Copy code
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    Add NVIDIA API Key:

3. **Create a .env file in the project root.
    ```Add your NVIDIA API Key:
    makefile
    Copy code
    NVIDIA_API_KEY=your_api_key_here
  
4. **Run the Application:
    ```bash
    Copy code
    streamlit run finalapp.py
    Upload Documents:

5. **Place your PDF files in the data/ directory.
Navigate to the "Embed Documents" tab in the app to start embedding.
Query Documents:

Go to the "Query" tab and enter your query to retrieve answers based on embedded content.


🌟 Usage Instructions
Home Tab
View an overview of the application and the RAG architecture.
Embed Documents Tab
Upload PDF documents to the data/ folder.
Click "Start Embedding" to preprocess and vectorize the documents.
Query Tab
Input queries to retrieve contextual answers from the embedded documents.
View detailed response times and document chunks retrieved during the query.


📋 Prerequisites
Python 3.9+
NVIDIA API Key for language model access
Streamlit for UI rendering
⚙️ Configuration
Ensure the following environment variable is set:

NVIDIA_API_KEY: Your API key for accessing NVIDIA's language models.
🔗 Resources
Streamlit Documentation
NVIDIA AI Models
LangChain Framework
