import streamlit as st
import pandas as pd
import json
from agent import create_agent, query_agent
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
import os

# ... (keep the existing functions: decode_response, write_response)

def update_faiss_vectorstore(file_path, file_type):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"}
    )

    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_type in ['doc', 'docx']:
        loader = Docx2txtLoader(file_path)
    else:
        st.error("Invalid file type. Please upload a PDF or Word Doc file.")
        return None

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    if not os.path.exists("faiss"):
        os.makedirs("faiss")

    try:
        db = FAISS.load_local("faiss", embeddings)
        db.add_documents(texts)
    except:
        db = FAISS.from_documents(texts, embeddings)
    
    db.save_local("faiss")
    print("FAISS updated")
    return db

st.title("Document Analysis App")

st.write("Please upload your PDF or Word Doc file below.")

uploaded_file = st.file_uploader("Upload a PDF or Word Doc file", type=["pdf", "doc", "docx"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner("Processing your document..."):
        # Save the uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Update the Faiss vectorstore with the uploaded file
        db = update_faiss_vectorstore(uploaded_file.name, file_type)
        
    
    st.success("Document processed successfully!")

query = st.text_area("Enter your query about the document")

if st.button("Submit Query", type="primary"):
    if 'db' not in locals() or db is None:
        st.error("Please upload a document first.")
    else:
        with st.spinner("Analyzing..."):
            # Create an agent from the vectorstore
            agent = create_agent(db)

            # Query the agent
            response = query_agent(agent=agent, query=query)

            # Display the response
            st.write("Response:")
            st.write(response)
            print("done")