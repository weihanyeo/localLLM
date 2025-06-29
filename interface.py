import os
import time
import uuid
import streamlit as st
from pathlib import Path
from datetime import datetime
import hashlib

# Chat utilities and agent functions
from chat_utils import save_conversation, load_conversation, list_conversations
from agent import create_agent, initialize_vectorstore, DocumentProcessor, query_agent

# Configuration
st.set_page_config(page_title="Document Analysis Chat", page_icon="💬", layout="wide")

# Constants
ASSETS_DIR = "./assets"
PAST_CONVO_DIR = "./PastConvo"
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(PAST_CONVO_DIR, exist_ok=True)

# Session Initialization
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.loaded_conversation = None

    with st.spinner("Initializing document index..."):
        db = initialize_vectorstore()
        if db is None:
            st.sidebar.warning("No documents found. Please upload documents to begin.")

# Sidebar - Document Upload & Conversation Control
with st.sidebar:
    st.header("Document Management")

    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Supported formats: PDF, Word, Text",
        type=["pdf", "doc", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        processor = DocumentProcessor()
        for uploaded_file in uploaded_files:
            file_path = os.path.join(ASSETS_DIR, uploaded_file.name)

            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
            existing_hash = (
                processor._get_file_hash(file_path)
                if os.path.exists(file_path) and hasattr(processor, '_get_file_hash')
                else None
            )

            if existing_hash == file_hash:
                st.warning(f"{uploaded_file.name} already exists with the same content.")
                continue
            elif existing_hash:
                st.warning(f"{uploaded_file.name} exists but differs. Overwriting...")

            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    if processor.process_new_file(file_path):
                        st.success(f"{uploaded_file.name} successfully indexed.")
                        st.experimental_rerun()
                    else:
                        st.error(f"Processing failed for {uploaded_file.name}.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.subheader("Past Conversations")
    past_convos = list_conversations()
    if past_convos:
        options = ["-- Select a conversation --"] + [
            f"{convo['created_at']} ({convo['message_count']} messages)"
            for convo in past_convos
        ]
        selected = st.selectbox("Choose", options)

        if selected != "-- Select a conversation --" and st.button("Load Conversation"):
            idx = options.index(selected) - 1
            convo = load_conversation(past_convos[idx]['session_id'])
            if convo and 'messages' in convo:
                st.session_state.chat_history = convo['messages']
                st.session_state.loaded_conversation = past_convos[idx]['session_id']
                st.experimental_rerun()

    if st.button("Start New Conversation"):
        if st.session_state.chat_history:
            save_conversation(
                st.session_state.chat_history,
                f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            )
        st.session_state.chat_history = []
        st.session_state.loaded_conversation = None
        st.session_state.session_id = str(uuid.uuid4())
        st.experimental_rerun()

# Main Interface - Chat Panel
st.title("💬 Document Analysis Chat")
st.write("Upload your documents and ask questions based on their content.")

st.subheader("Conversation")

# Display Chat History
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        sources = msg.get("sources", [])

        cols = st.columns([1, 5])
        with cols[0]:
            st.markdown("**👤 You**" if role == "user" else "**🤖 Assistant**")
        with cols[1]:
            if timestamp:
                st.markdown(f"*{timestamp}*")
            st.markdown(content)

            if role != "user" and sources:
                with st.expander("View Sources"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Source {i}** — {src.get('file', 'Unknown')} (Page {src.get('page', 'N/A')})")
                        st.code(src.get("content", ""), language="markdown")
        st.markdown("---")

# Chat Input
with st.form("chat_form", clear_on_submit=True):
    prompt = st.text_area("Ask a question based on your documents", height=100)
    submitted = st.form_submit_button("Send")

# Handle Submission
if submitted and prompt.strip():
    st.session_state.chat_history.append({"role": "user", "content": prompt.strip()})

    with st.spinner("Generating response..."):
        time_start = time.time()
        db = initialize_vectorstore()

        if db:
            agent = create_agent(db, st.session_state.chat_history)
            response = query_agent(
                agent=agent,
                query=prompt,
                chat_history=st.session_state.chat_history,
                session_id=st.session_state.session_id
            )

            st.session_state.chat_history = response["chat_history"]
            save_conversation(st.session_state.chat_history, st.session_state.session_id)

            time_elapsed = time.time() - time_start
            st.sidebar.info(f"Response time: {time_elapsed:.2f}s")

            st.experimental_rerun()
        else:
            st.error("No documents indexed. Please upload documents first.")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "I couldn't find any documents to analyze. Please upload some first."
            })

# Auto-save on exit (placeholder only; real implementation needs backend endpoint)
st.markdown(
    """
    <script>
    window.addEventListener('beforeunload', function() {
        fetch('/_stcore/save-conversation', { method: 'POST' });
    });
    </script>
    """,
    unsafe_allow_html=True
)
