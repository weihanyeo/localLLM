import streamlit as st
from agent import create_agent, query_agent
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
import os

# Initialize embeddings
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large",
    model_kwargs={"device": "cpu"}
)

# Create ./faiss directory if it doesn't exist
if not os.path.exists("./faiss"):
    os.makedirs("./faiss", exist_ok=True)

# Create ./assets directory if it doesn't exist
if not os.path.exists("./assets"):
    os.makedirs("./assets", exist_ok=True)

# Try to load existing FAISS index
try:
    if os.path.exists("faiss/index.faiss"):
        db = FAISS.load_local("faiss", embeddings)
        st.success("Existing document database loaded successfully!")
    else:
        st.info("No existing document database found. Please upload a document to get started.")
except Exception as e:
    st.error(f"Error loading existing database: {str(e)}")
    st.info("Starting with a fresh database. Please upload a document to get started.")

def update_faiss_vectorstore(file_path):
    global embeddings
    
    all_texts = []
    for file_path in file_paths:
        file_type = file_path.split('.')[-1].lower()
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_type in ['doc', 'docx']:
            loader = Docx2txtLoader(file_path)
        else:
            st.error(f"Invalid file type for {file_path}. Skipping.")
            continue

        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
        texts = splitter.split_documents(documents)
        all_texts.extend(texts)

    try:
        # Try to load the existing FAISS index from the local file
        db = FAISS.load_local("faiss", embeddings)
        db.add_documents(texts)
    except FileNotFoundError:
        # If the file is not found, create a new FAISS index from the documents
        db = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        # Catch any other exceptions and handle accordingly
        print(f"An error occurred: {e}")
        db = FAISS.from_documents(texts, embeddings)

    db.save_local("faiss")
    print("FAISS updated")
    return db

st.title("Document Analysis App")

st.write("Please upload your PDF or Word Doc file below.")

uploaded_files = st.file_uploader("Upload a PDF or Word Doc file", type=["pdf", "doc", "docx"], accept_multiple_files=True)

if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join("./assets", uploaded_file.name)
        
        with st.spinner("Processing {uploaded_file.name} document..."):
            # Save the uploaded file temporarily
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
            st.success(f"File {uploaded_file.name} saved to ./assets folder")
            
    with st.spinner("Vectorising contents..."):
        db = update_faiss_vectorstore(file_paths)

    if db is not None:
        st.success("All documents processed successfully!")
    else:
        st.error("Failed to process the documents. Please try again.")
        
    st.success("Document processed successfully!")

query = st.text_area("Enter your query about the document")

if st.button("Submit Query", type="primary"):
    if db is None:
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