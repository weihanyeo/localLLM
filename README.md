# Document Analysis Chat

A powerful document analysis tool that allows you to chat with your documents using local language models. Built with LangChain, Streamlit, and FAISS for efficient document processing and retrieval.

## 🌟 Features

- **Document Processing**: Supports PDF, Word (.doc, .docx), and Text (.txt) formats
- **Efficient Indexing**: Uses FAISS for fast similarity search
- **Conversational Memory**: Saves and loads previous conversations
- **Local Processing**: All processing happens on your machine
- **Source Attribution**: Shows which parts of documents were used to generate answers

## 🚀 Quick Start

### Prerequisites

- **Hardware Requirements**:
  - Minimum RAM: 8GB (16GB recommended for larger documents)
  - CPU: 4+ cores (10-12 cores recommended for better performance)

- **Software Requirements**:
  - Python 3.8+
  - Anaconda (recommended for environment management)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/localLLM.git
   cd localLLM
   ```

2. **Create and activate a conda environment**:
   ```bash
   conda create -n docchat python=3.8
   conda activate docchat
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

1. **Start the application**:
   ```bash
   streamlit run interface.py
   ```
   This will open the application in your default browser at [http://localhost:8501/](http://localhost:8501/)

2. **Upload documents**:
   - Click on the upload button in the sidebar
   - Select one or more PDF, Word, or Text files
   - Wait for the documents to be processed (you'll see a success message)

3. **Start chatting**:
   - Type your question in the chat box
   - Press Enter or click "Send"
   - The AI will analyze your documents and provide answers with sources

## 🏗️ Architecture

The application consists of three main components:

1. **interface.py**: Streamlit-based web interface for user interaction
2. **agent.py**: Core document processing and question-answering logic
3. **chat_utils.py**: Conversation history management

### Data Flow

1. **Document Ingestion**:
   - User uploads documents through the web interface
   - Documents are processed and split into chunks
   - Chunks are converted to vector embeddings using HuggingFace's `all-MiniLM-L12-v2`
   - Vectors are indexed using FAISS for efficient similarity search

2. **Question Answering**:
   - User submits a question
   - Question is converted to a vector
   - Most relevant document chunks are retrieved
   - Local LLM generates an answer based on the retrieved context
   - Response is displayed with source attribution

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Path to store document assets
ASSETS_DIR=./assets

# Path to store FAISS index
FAISS_INDEX=./FAISS

# Path to store conversation history
PAST_CONVO_DIR=./PastConvo
```

### Model Configuration

The default model configuration can be modified in `agent.py`:

```python
# In DocumentProcessor.__init__
self.embeddings = HuggingFaceInstructEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}  # Change to "cuda" if you have a GPU
)
```

## 📚 Module Reference

### agent.py

Core document processing and question-answering functionality.

#### Key Classes & Functions:

- **DocumentProcessor**: Handles document loading, processing, and vector storage
  - `process_document()`: Process a single document into chunks
  - `update_vectorstore()`: Update the FAISS index with new documents
  - `file_needs_processing()`: Check if a file needs reprocessing

- **create_agent()**: Initialize a conversational agent with document context
- **query_agent()**: Process a user query and return a formatted response

### chat_utils.py

Manages conversation history and persistence.

#### Key Functions:

- `save_conversation()`: Save chat history to disk
- `load_conversation()`: Load a previous conversation
- `list_conversations()`: List available conversation history
- `delete_conversation()`: Remove a saved conversation

### interface.py

Streamlit-based web interface for the application.

#### Key Features:
- Document upload and processing
- Interactive chat interface
- Conversation history management
- Source attribution display

## 🛠️ Troubleshooting

### Common Issues

1. **Slow Performance**:
   - Ensure you have enough free RAM
   - Close other memory-intensive applications
   - Process documents in smaller batches

2. **Document Processing Failures**:
   - Check file formats (PDF, DOCX, DOC, TXT only)
   - Ensure files are not password protected
   - Verify file integrity

3. **Model Loading Issues**:
   - Check internet connection (required for first-time model download)
   - Verify sufficient disk space for model weights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [HuggingFace](https://huggingface.co/) for the embedding models
- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
