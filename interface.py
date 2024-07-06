import streamlit as st
from agent import create_agent, query_agent
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
import os

# Create ./assets directory if it doesn't exist
if not os.path.exists("./assets"):
    os.makedirs("./assets", exist_ok=True)

def update_faiss_vectorstore():
    # Initialize embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base",
        model_kwargs={"device": "cpu"}
    )
    
    all_texts = []
    directory = "./assets"
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        file_type = filename.split('.')[-1].lower()
        
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_type in ['doc', 'docx']:
            loader = Docx2txtLoader(file_path)
        elif file_type == 'txt':
            loader = TextLoader(file_path)
        else:
            st.warning(f"Skipping unsupported file: {filename}")
            continue

        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(documents)
        all_texts.extend(texts)

    db = FAISS.from_documents(all_texts, embeddings)
    return db

st.title("Document Analysis App")

st.write("Please upload your PDF, Word Doc, or Text file below.")

uploaded_files = st.file_uploader("Upload a PDF, Word Doc, or Text file", type=["pdf", "doc", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("./assets", uploaded_file.name)
        
        with st.spinner(f"Saving {uploaded_file.name}..."):
            # Save the uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File {uploaded_file.name} saved to ./assets folder")

query = st.text_area("Enter your query about the document")

if st.button("Submit Query", type="primary"):
    with st.spinner("Vectorizing contents and analyzing..."):
        # Vectorising all uploads in ./assets
        db = update_faiss_vectorstore()
        
        if db is not None:
            # Create an agent from the vectorstore
            agent = create_agent(db)

            # Query the agent
            response = query_agent(agent=agent, query=query)

            # Display the response
            st.write("Response:")
            st.write(response)
        else:
            st.error("Failed to process the documents. Please make sure you have uploaded at least one supported file.")