import streamlit as st
import os
import time
from pathlib import Path
import pandas as pd
from typing import List, Dict
import tempfile

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import our RAG components
from src.rag_chain import QueryProcessor
from src.vector_store import VectorStore
from src.document_processor import DocumentProcessor
from src.document_loader import PDFDocumentLoader
from config import GROQ_API_KEY

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Q&A with RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    
    LEARNING NOTE: Session state maintains data across Streamlit reruns
    - Keeps conversation history
    - Stores uploaded documents
    - Maintains system configuration
    """
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'query_processor' not in st.session_state:
        st.session_state.query_processor = None
    
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

def load_default_documents():
    """Load sample documents if no custom documents are uploaded."""
    try:
        # Try to load existing vector store
        vector_store = VectorStore()
        vector_store.load_vector_store("data/vector_store")
        return vector_store, "Loaded existing sample documents"
    except FileNotFoundError:
        # Create sample documents
        loader = PDFDocumentLoader()
        documents = loader.create_sample_documents()
        
        processor = DocumentProcessor()
        chunks = processor.process_documents(documents)
        
        vector_store = VectorStore()
        vector_store.create_vector_store(chunks)
        vector_store.save_vector_store("data/vector_store")
        
        return vector_store, "Created sample documents"

def process_uploaded_files(uploaded_files, chunk_size=1000, chunk_overlap=200):
    """
    Process uploaded PDF files.
    
    LEARNING NOTE: This handles the full document processing pipeline:
    1. Save uploaded files temporarily
    2. Extract text from PDFs
    3. Process into chunks
    4. Create vector store
    """
    if not uploaded_files:
        return None, "No files uploaded"
    
    loader = PDFDocumentLoader()
    all_documents = []
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        
        try:
            # Load document from PDF
            pdf_bytes = uploaded_file.getvalue()
            document = loader.load_from_pdf_bytes(pdf_bytes, uploaded_file.name)
            
            if document:
                all_documents.append(document)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            continue  # Skip this file, continue with others
    
    if not all_documents:
        return None, "No valid documents found"
    
    # Process documents into chunks
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = processor.process_documents(all_documents)
    
    # Create vector store
    vector_store = VectorStore()
    vector_store.create_vector_store(chunks)
    
    return vector_store, f"Processed {len(all_documents)} documents into {len(chunks)} chunks"

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("📚 PDF Q&A with RAG")
    st.markdown("Upload PDFs and ask questions about their content using AI!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        if GROQ_API_KEY:
            st.success("✅ Groq API Key loaded")
        else:
            st.error("❌ Groq API Key not found")
            st.info("Please add your Groq API key to config.py")
            return
        
        # Model selection
        model_options = {
            "llama-3.3-70b-versatile": "Llama 3.3 70B (Recommended)",
            "llama-3.1-8b-instant": "llama-3.1-8b-instant (Fastest)"
        }
        
        selected_model = st.selectbox(
            "🤖 Select Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            help="Choose the AI model for generating answers"
        )
        
        # Handle model changes
        if 'current_model' not in st.session_state:
            st.session_state.current_model = selected_model

        if st.session_state.current_model != selected_model:
            # Model changed - reset query processor to force recreation
            st.session_state.query_processor = None
            st.session_state.current_model = selected_model
            
            # Show user feedback
            model_name = model_options[selected_model]
            st.info(f"🔄 Switched to {model_name}")
            
            # If documents are already processed, the query processor will be recreated automatically
            if st.session_state.documents_processed:
                st.info("Next question will use the new model")

        # Document processing settings
        st.subheader("📄 Document Settings")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100, 
                              help="Size of text chunks for processing")
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50,
                                 help="Overlap between chunks for context")
        retrieval_k = st.slider("Retrieved Chunks", 1, 10, 3, 1,
                               help="Number of relevant chunks to retrieve")
        
        # Document upload
        st.subheader("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDF documents to ask questions about"
        )
        
        # Process documents button
        if st.button("🔄 Process Documents", type="primary"):
            if uploaded_files:
                with st.spinner("Processing uploaded documents..."):
                    st.session_state.vector_store = None
                    st.session_state.query_processor = None
                    st.session_state.documents_processed = False
                    st.session_state.messages = [] 
                            
                    vector_store, message = process_uploaded_files(
                        uploaded_files, chunk_size, chunk_overlap
                    )
                    
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.documents_processed = True
                        st.session_state.uploaded_files = [f.name for f in uploaded_files]
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.warning("Please upload PDF files first")
        
        # Load sample documents button
        if st.button("📚 Load Sample Documents"):
            with st.spinner("Loading sample documents..."):
                vector_store, message = load_default_documents()
                st.session_state.vector_store = vector_store
                st.session_state.documents_processed = True
                st.session_state.uploaded_files = ["Sample Documents"]
                st.success(message)
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Initialize query processor if documents are processed
        if (st.session_state.documents_processed and 
            st.session_state.vector_store and 
            not st.session_state.query_processor):
            
            try:
                query_processor = QueryProcessor(GROQ_API_KEY, selected_model)
                query_processor.load_vector_store(st.session_state.vector_store)
                st.session_state.query_processor = query_processor
                st.success("🚀 RAG system ready!")
            except Exception as e:
                st.error(f"Error initializing RAG system: {e}")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**{i}. {source['title']}**")
                            st.markdown(f"*Similarity: {source['similarity_score']:.3f}*")
                            st.markdown(f"```\n{source['content_preview']}\n```")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.query_processor:
                st.error("Please process documents first using the sidebar")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = st.session_state.query_processor.process_query(
                            prompt, k=retrieval_k
                        )
                    
                    # Display answer
                    st.markdown(result["answer"])
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", [])
                    })
    
    with col2:
        st.header("📊 System Status")
        
        if st.session_state.vector_store:
            info = st.session_state.vector_store.get_store_info()
            
            # Display system metrics
            st.metric("📄 Total Chunks", info['total_chunks'])
            st.metric("📚 Unique Sources", info['unique_sources'])
            st.metric("🔢 Vector Dimensions", info['vector_dimensions'])
            
            # Display processed files
            st.subheader("📁 Processed Files")
            for filename in st.session_state.uploaded_files:
                st.text(f"• {filename}")
        
        else:
            st.info("No documents processed yet")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()