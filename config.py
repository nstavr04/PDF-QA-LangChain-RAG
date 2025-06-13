import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, 90MB download
LLM_MODEL = "llama3-8b-8192"  # Groq's fast Llama model

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_RETRIEVAL_DOCS = 4

# Vector Store Configuration
VECTOR_STORE_PATH = "data/vector_store"

# PDF Processing Configuration
MAX_PDF_SIZE_MB = 50  # Maximum PDF file size
SUPPORTED_FORMATS = ['.pdf', '.txt']