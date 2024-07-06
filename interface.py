import streamlit as st
from agent import create_agent, query_agent
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
import os
import time

# Create ./assets directory if it doesn't exist
if not os.path.exists("./assets"):
    os.makedirs("./assets", exist_ok=True)

def update_faiss_vectorstore():
    # Initialize embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
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
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
        texts = splitter.split_documents(documents)
        all_texts.extend(texts)

    db = FAISS.from_documents(all_texts, embeddings)
    st.success("Vector store updated sucessfully")
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
            time_start = time.time()
            # Create an agent from the vectorstore
            agent = create_agent(db)
            # Query the agent
            response = query_agent(agent=agent, query=query)

            # Display question and answer
            st.markdown(f"**Question:** {query}")
            st.markdown(f"**Answer:** {response}")

            # Display sources if available
            if 'source_documents' in response:
                st.markdown("**Sources:**")
                for i, doc in enumerate(response['source_documents'], 1):
                    st.markdown(f"{i}. {doc.metadata.get('thisssource', 'Unknown source')}")
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

            # Stops and Display timing
            time_elapsed = time.time() - time_start 
            st.code(f"Response time: {time_elapsed:.02f} sec")
        else:
            st.error("Failed to process the documents. Please make sure you have uploaded at least one supported file.")