import os
import hashlib
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from accelerate import Accelerator
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Import chat utilities
from chat_utils import save_conversation, load_conversation, list_conversations

accelerator = Accelerator()

# Constants
ASSETS_DIR = "./assets"
FAISS_INDEX = "./FAISS"
FILE_HASHES = os.path.join(ASSETS_DIR, ".file_hashes.json")

class DocumentProcessor:
    def __init__(self):
        self.embeddings = HuggingFaceInstructEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"}
        )
        self.file_hashes = self._load_file_hashes()
        
    def _load_file_hashes(self) -> Dict[str, str]:
        """Load file hashes from the JSON file."""
        import json
        if os.path.exists(FILE_HASHES):
            with open(FILE_HASHES, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_file_hashes(self):
        """Save file hashes to the JSON file."""
        import json
        with open(FILE_HASHES, 'w') as f:
            json.dump(self.file_hashes, f)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def file_needs_processing(self, file_path: str) -> bool:
        """Check if a file needs to be processed (new or modified)."""
        if not os.path.exists(file_path):
            return False
            
        file_hash = self._get_file_hash(file_path)
        file_name = os.path.basename(file_path)
        
        if file_name in self.file_hashes:
            if self.file_hashes[file_name] == file_hash:
                return False  # File hasn't changed
        
        # Update the hash for this file
        self.file_hashes[file_name] = file_hash
        return True
    
    def process_document(self, file_path: str) -> List[Document]:
        """Process a single document and return its chunks."""
        file_type = os.path.splitext(file_path)[1][1:].lower()
        
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_type in ['doc', 'docx']:
            loader = Docx2txtLoader(file_path)
        elif file_type == 'txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased chunk size for better context
            chunk_overlap=100,
            length_function=len
        )
        return splitter.split_documents(documents)
    
    def update_vectorstore(self, force_update: bool = False) -> Optional[FAISS]:
        """
        Update the FAISS vector store with new or modified documents.
        
        Args:
            force_update: If True, forces reindexing of all documents
            
        Returns:
            FAISS: The updated FAISS index, or None if no documents found
        """
        if not os.path.exists(ASSETS_DIR):
            os.makedirs(ASSETS_DIR, exist_ok=True)
            return None
        
        # Check for new or modified files
        files_to_process = []
        for filename in os.listdir(ASSETS_DIR):
            if filename.startswith('.'):  # Skip hidden files and our hash file
                continue
                
            file_path = os.path.join(ASSETS_DIR, filename)
            if force_update or self.file_needs_processing(file_path):
                files_to_process.append(file_path)
        
        # If no files to process and index exists, return existing index
        if not files_to_process and os.path.exists(FAISS_INDEX):
            try:
                return FAISS.load_local(FAISS_INDEX, self.embeddings)
            except Exception as e:
                print(f"Error loading existing index, will create new one: {str(e)}")
                files_to_process = [
                    os.path.join(ASSETS_DIR, f) for f in os.listdir(ASSETS_DIR)
                    if not f.startswith('.')
                ]
        
        # Process all documents
        all_docs = []
        for file_path in files_to_process:
            try:
                print(f"Processing file: {file_path}")
                docs = self.process_document(file_path)
                all_docs.extend(docs)
                # Update the hash for successfully processed files
                if not force_update:
                    self.file_hashes[os.path.basename(file_path)] = self._get_file_hash(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # If no documents were processed and no index exists, return None
        if not all_docs and not os.path.exists(FAISS_INDEX):
            return None
            
        # If no new documents but index exists, return existing index
        if not all_docs and os.path.exists(FAISS_INDEX):
            return FAISS.load_local(FAISS_INDEX, self.embeddings)
        
        # Create or update the FAISS index
        try:
            if os.path.exists(FAISS_INDEX):
                db = FAISS.load_local(FAISS_INDEX, self.embeddings)
                if all_docs:  # Only add new documents if there are any
                    print(f"Adding {len(all_docs)} new document chunks to existing index")
                    db.add_documents(all_docs)
                    db.save_local(FAISS_INDEX)
            else:  # Create new index
                print(f"Creating new index with {len(all_docs)} document chunks")
                db = FAISS.from_documents(all_docs, self.embeddings)
                db.save_local(FAISS_INDEX)
            
            # Save the updated file hashes
            self._save_file_hashes()
            return db
            
        except Exception as e:
            print(f"Error updating FAISS index: {str(e)}")
            # If there's an error, try to create a new index
            if all_docs and os.path.exists(FAISS_INDEX):
                try:
                    print("Attempting to create new index after error")
                    db = FAISS.from_documents(all_docs, self.embeddings)
                    db.save_local(FAISS_INDEX)
                    self._save_file_hashes()
                    return db
                except Exception as e2:
                    print(f"Failed to create new index: {str(e2)}")
            return None
    
    def process_new_file(self, file_path: str) -> bool:
        """
        Process a single new file and update the vector store.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
            
        try:
            print(f"Processing new file: {file_path}")
            # Process the document
            docs = self.process_document(file_path)
            if not docs:
                print(f"No content extracted from {file_path}")
                return False
                
            # Update the hash
            self.file_hashes[os.path.basename(file_path)] = self._get_file_hash(file_path)
            
            # Update the vector store
            if os.path.exists(FAISS_INDEX):
                db = FAISS.load_local(FAISS_INDEX, self.embeddings)
                db.add_documents(docs)
                db.save_local(FAISS_INDEX)
            else:
                db = FAISS.from_documents(docs, self.embeddings)
                db.save_local(FAISS_INDEX)
                
            # Save the updated hashes
            self._save_file_hashes()
            print(f"Successfully processed and indexed {file_path}")
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return False

def initialize_vectorstore() -> Optional[FAISS]:
    """Initialize and return the FAISS vector store."""
    processor = DocumentProcessor()
    return processor.update_vectorstore()

def create_agent(vectorstore, chat_history: Optional[List[Dict[str, str]]] = None):
    """
    Create a conversational agent with optional chat history.
    
    Args:
        vectorstore: The vector store for document retrieval
        chat_history: List of previous messages in the format [{"role": "user"|"assistant", "content": "message"}]
    """
    config = {
        'max_new_tokens': 512, 
        'temperature': 0.1, 
        'context_length': 2400,
        'threads': os.cpu_count()
    }

    llm = CTransformers(
        model="./models/llama/llama-2-7b-chat.ggmlv3.q3_K_M.bin",
        model_type="llama",
        config=config
    )

    # Prepare llm to accelerate
    llm, config = accelerator.prepare(llm, config)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

    # Build context from chat history
    chat_history_context = ""
    if chat_history:
        chat_history_context = "Previous conversation history:\n"
        for msg in chat_history[-5:]:  # Include last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_history_context += f"{role}: {msg['content']}\n"
        chat_history_context += "\n"

    template = f"""Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Determine the most appropriate answer and always refer and revisit the given documents.
    
    Chat History Context:{chat_history_context}
    Context from documents:{{context}}
    Question: {{question}}
    
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

    qa_llm = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_llm

def query_agent(agent, query: str, chat_history: List[Dict[str, Any]] = None, session_id: str = None) -> Dict[str, Any]:
    """
    Query the agent and format the response with conversation history.
    
    Args:
        agent: The agent to query
        query: The user's question
        chat_history: List of previous messages with metadata
        session_id: Optional session ID for saving the conversation
        
    Returns:
        Dict containing the response, sources, and updated chat history
    """
    if chat_history is None:
        chat_history = []
        
    try:
        # Add user message to chat history with timestamp
        user_message = {
            "role": "user", 
            "content": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        chat_history.append(user_message)
        
        # Get response from agent
        response = agent({"query": query})
        answer = response.get('result', 'No answer found.')
        sources = response.get('source_documents', [])
        
        # Format sources for display
        formatted_sources = []
        source_files = set()
        
        for doc in sources:
            file_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
            source_files.add(file_name)
            content = doc.page_content
            content_preview = (content[:750] + '...') if len(content) > 750 else content
            formatted_sources.append({
                'file': file_name,
                'content': content_preview,
                'page': doc.metadata.get('page', 'N/A')
            })
        
        # Include source files in the answer
        timestamp = datetime.now().strftime("%H:%M:%S")
        if source_files:
            sources_text = ", ".join(f'"{f}"' for f in source_files)
            answer = f"{answer}\n\n*Sources: {sources_text} | Generated at {timestamp}*"
        else:
            answer = f"{answer}\n\n*Generated at {timestamp}*"
        
        # Add assistant's response to history with metadata
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "timestamp": timestamp,
            "sources": formatted_sources
        }
        chat_history.append(assistant_message)
        
        # Save conversation if session_id is provided
        if session_id:
            save_conversation(chat_history, session_id)
        
        return {
            'answer': answer,
            'sources': formatted_sources,
            'chat_history': chat_history,
            'session_id': session_id or str(uuid.uuid4()),
            'timestamp': timestamp
        }
        
    except Exception as e:
        error_msg = f"Error querying the agent: {str(e)}"
        chat_history.append({"role": "assistant", "content": error_msg})
        return {
            'answer': error_msg,
            'sources': [],
            'chat_history': chat_history,
            'session_id': session_id or str(uuid.uuid4()),
            'error': str(e)
        }