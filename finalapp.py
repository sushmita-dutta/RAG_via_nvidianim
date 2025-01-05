import streamlit as st
import os
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Set up Streamlit layout (must be first Streamlit command)
st.set_page_config(
    page_title="Advanced RAG App",
    layout="wide",
    page_icon="📚"
)

# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

# Initialize NVIDIA LLM
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

# Initialize session state for vector embedding
def vector_embed():
    with st.spinner("Embedding documents into vector space..."):
        if "vectors" not in st.session_state:
            st.session_state.embeddings = NVIDIAEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./data")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700, chunk_overlap=100
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:20]
            )
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings
            )
        st.success("Document embedding complete!")

# CSS for styling
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            color: #4CAF50;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stMarkdown {
            font-size: 18px;
            margin: 20px 0;
        }
        .response-box {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info(
        """
        **Instructions**:
        - Ensure your NVIDIA API key is set up.
        - Add documents to the `./data` directory.
        - Use the buttons below to interact with the app.
        """
    )
    st.markdown("**Created by Sushmita Dutta** 💡")

# Main Title
st.markdown('<div class="main-title">📚 Advanced Retrieval-Augmented Generation (RAG) App</div>', unsafe_allow_html=True)

# Tabs for better organization
tabs = st.tabs(["Home", "Embed Documents", "Query"])

# Home Tab
with tabs[0]:
    st.subheader("Welcome to the RAG Application")
    st.markdown(
        """
        This application enables you to:
        - Embed PDF documents into a vector store for efficient retrieval.
        - Query the documents using NVIDIA's advanced language models.
        """
    )
    st.image("./RAG-Architecture.png", caption="RAG Architecture Overview", use_container_width=True)

# Embed Documents Tab
with tabs[1]:
    st.subheader("📂 Embed Documents into Vector Store")
    st.markdown(
        """
        Upload your documents in the `./data` directory, then click the button below to start the embedding process.
        """
    )
    if st.button("Start Embedding"):
        vector_embed()

# Query Tab
with tabs[2]:
    st.subheader("🔍 Query Your Documents")
    prompt_user = st.text_input(
        "Enter your query below:",
        placeholder="E.g., What is the summary of document X?"
    )

    if "vectors" in st.session_state:
        if prompt_user:
            st.markdown("#### 📝 Query Response")
            with st.spinner("Generating response..."):
                prompt = ChatPromptTemplate.from_template(
                    """
                    Answer the questions based on the provided context only.
                    Please provide the most accurate response based on the question.
                    <context>
                    {context}
                    <context>
                    Question: {input}
                    """
                )
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({"input": prompt_user})
                response_time = time.process_time() - start

                st.markdown('<div class="response-box">', unsafe_allow_html=True)
                st.write(f"**Response:** {response['answer']}")
                st.write(f"⏱️ **Response Time:** {response_time:.2f} seconds")
                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("🔍 View Retrieved Document Chunks"):
                    for i, doc in enumerate(response['context']):
                        st.write(f"**Chunk {i + 1}:**")
                        st.write(doc.page_content)
                        st.write("---")
        else:
            st.warning("⚠️ Please enter a query to get started!")
    else:
        st.warning("⚠️ Embed your documents first to enable querying!")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; font-size: 14px; color: gray;">
        © 2025 Sushmita Dutta | Powered by NVIDIA NIM and Streamlit
    </div>
    """, 
    unsafe_allow_html=True
)
