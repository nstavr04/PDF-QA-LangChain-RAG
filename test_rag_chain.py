from src.rag_chain import QueryProcessor
from src.vector_store import VectorStore
from config import GROQ_API_KEY

def demo_rag_system():
    """
    Demo the complete RAG system with interactive chat.
    
    LEARNING NOTE: This shows the end-to-end user experience
    - Load documents and create vector store
    - Start interactive Q&A session
    - Users can ask questions and get intelligent answers
    """
    print("🚀 PDF Q&A System Demo")
    print("="*40)
    
    if not GROQ_API_KEY:
        print("❌ Please set GROQ_API_KEY in config.py")
        return
    
    # Initialize system
    print("🔄 Initializing RAG system...")
    
    # Load vector store
    vector_store = VectorStore()
    try:
        vector_store.load_vector_store("data/vector_store")
        print("✅ Vector store loaded")
    except FileNotFoundError:
        print("❌ Vector store not found. Please run Phase 4 first.")
        return
    
    # Initialize query processor
    query_processor = QueryProcessor(GROQ_API_KEY)
    query_processor.load_vector_store(vector_store)
    
    # Show system info
    info = vector_store.get_store_info()
    print(f"📊 System ready: {info['total_chunks']} chunks from {info['unique_sources']} sources")
    
    # Start interactive chat
    query_processor.interactive_chat()

if __name__ == "__main__":
    demo_rag_system()