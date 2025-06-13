from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Dict, List, Optional, Tuple
import logging
import os
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize vector store with embedding model.
        
        LEARNING NOTE: What are embeddings?
        - Embeddings convert text into vectors (lists of numbers)
        - Similar text gets similar vectors
        - We can then use math to find similar content
        - "sentence-transformers/all-MiniLM-L6-v2" is free, fast, and good quality
        
        Args:
            embedding_model_name: HuggingFace model for creating embeddings
        """
        self.embedding_model_name = embedding_model_name
        self.vector_store = None
        self.chunks = []  # Keep original chunks for metadata
        
        # LEARNING NOTE: This model converts text to 384-dimensional vectors
        # Each dimension captures different aspects of meaning
        print(f"🔄 Loading embedding model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},  # Use CPU (free!)
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
        )
        print("✅ Embedding model loaded successfully")
    
    def create_vector_store(self, chunks: List[Dict]) -> None:
        """
        Create FAISS vector store from document chunks.
        
        LEARNING NOTE: This is where the magic happens!
        1. Each chunk gets converted to a vector (embedding)
        2. All vectors get stored in FAISS for fast similarity search
        3. Original text and metadata are preserved for retrieval
        
        Args:
            chunks: List of chunk dictionaries with 'content' and 'metadata'
        """
        if not chunks:
            raise ValueError("No chunks provided for vector store creation")
        
        print(f"🔄 Creating vector store from {len(chunks)} chunks...")
        
        # Extract text content for embedding
        # LEARNING NOTE: We only embed the text content, not metadata
        texts = [chunk['content'] for chunk in chunks]
        
        # Extract metadata for each chunk
        # LEARNING NOTE: Metadata helps us trace answers back to sources
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Create FAISS vector store
        # LEARNING NOTE: This does several things:
        # 1. Converts each text to embeddings using our model
        # 2. Stores embeddings in FAISS index for fast search
        # 3. Links each embedding to its original text and metadata
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # Keep original chunks for reference
        self.chunks = chunks
        
        print(f"✅ Vector store created with {len(chunks)} embeddings")
        print(f"📊 Vector dimensions: {len(self.embeddings.embed_query('test'))}")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, Dict, float]]:
        """
        Search for similar chunks using vector similarity.
        
        LEARNING NOTE: How similarity search works:
        1. Convert user query to vector using same embedding model
        2. Calculate similarity between query vector and all stored vectors
        3. Return top K most similar chunks
        4. Similarity is measured using cosine similarity (angle between vectors)
        
        Args:
            query: User's question/query
            k: Number of similar chunks to return
            
        Returns:
            List of (content, metadata, similarity_score) tuples
        """
        if not self.vector_store:
            raise ValueError("Vector store not created. Call create_vector_store() first.")
        
        print(f"🔍 Searching for: '{query}' (returning top {k} results)")
        
        # LEARNING NOTE: This does the vector math!
        # 1. query → embedding vector
        # 2. Compare with all stored vectors
        # 3. Return most similar ones
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Format results for easier use
        formatted_results = []
        for doc, score in results:
            formatted_results.append((
                doc.page_content,  # The chunk text
                doc.metadata,      # Source information
                score             # Similarity score (lower = more similar)
            ))
        
        print(f"✅ Found {len(formatted_results)} similar chunks")
        return formatted_results
    
    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """
        Get relevant context for a query as a single string.
        
        LEARNING NOTE: This prepares context for the LLM
        - Combines multiple relevant chunks
        - Includes source attribution
        - Ready to be fed to GPT/Claude for answer generation
        
        Args:
            query: User's question
            k: Number of chunks to include
            
        Returns:
            Combined context string with source attribution
        """
        results = self.similarity_search(query, k=k)
        
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, (content, metadata, score) in enumerate(results, 1):
            source = metadata.get('title', 'Unknown Source')
            chunk_info = f"(Chunk {metadata.get('chunk_id', 0)+1}/{metadata.get('chunk_count', 1)})"
            
            context_parts.append(f"--- Source {i}: {source} {chunk_info} ---\n{content}")
        
        return "\n\n".join(context_parts)
    
    def save_vector_store(self, save_path: str) -> None:
        """
        Save vector store to disk.
        
        LEARNING NOTE: Why save?
        - Creating embeddings is slow and expensive
        - Once created, we can reuse the vector store
        - Saves time on subsequent runs
        """
        if not self.vector_store:
            raise ValueError("No vector store to save")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.vector_store.save_local(str(save_path))
        
        # Save chunks separately
        with open(save_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        print(f"💾 Vector store saved to {save_path}")
    
    def load_vector_store(self, load_path: str) -> None:
        """
        Load vector store from disk.
        
        LEARNING NOTE: This lets us skip the embedding creation step
        if we've already processed these documents before.
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        # Load FAISS index
        self.vector_store = FAISS.load_local(
            str(load_path), 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Load chunks
        with open(load_path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        
        print(f"📂 Vector store loaded from {load_path}")
    
    def get_store_info(self) -> Dict:
        """Get information about the vector store."""
        if not self.vector_store:
            return {"status": "not_created"}
        
        return {
            "status": "ready",
            "total_chunks": len(self.chunks),
            "embedding_model": self.embedding_model_name,
            "vector_dimensions": len(self.embeddings.embed_query("test")),
            "unique_sources": len(set(chunk['metadata'].get('title', 'Unknown') 
                                   for chunk in self.chunks))
        }

# Test function
def test_vector_store():
    """
    Test the complete vector store pipeline.
    
    LEARNING NOTE: This demonstrates the full Phase 4 process:
    1. Load processed chunks from Phase 3
    2. Create embeddings and vector store
    3. Test similarity search
    4. Show how retrieval works
    """
    from document_processor import DocumentProcessor
    from document_loader import PDFDocumentLoader
    
    print("🧪 Testing Vector Store Pipeline...")
    
    # Step 1: Get processed chunks (from Phase 3)
    print("\n1. Loading and processing documents...")
    loader = PDFDocumentLoader()
    documents = loader.create_sample_documents()
    
    processor = DocumentProcessor()
    chunks = processor.process_documents(documents)
    print(f"✅ Got {len(chunks)} processed chunks")
    
    # Step 2: Create vector store
    print("\n2. Creating vector store...")
    vector_store = VectorStore()
    vector_store.create_vector_store(chunks)
    
    # Step 3: Show store info
    print("\n3. Vector store information:")
    info = vector_store.get_store_info()
    for key, value in info.items():
        print(f"   📊 {key.replace('_', ' ').title()}: {value}")
    
    # Step 4: Test similarity search
    print("\n4. Testing similarity search...")
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is backpropagation?",
        "Explain artificial intelligence"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = vector_store.similarity_search(query, k=2)
        
        for i, (content, metadata, score) in enumerate(results, 1):
            source = metadata.get('title', 'Unknown')
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"   {i}. [{source}] Score: {score:.3f}")
            print(f"      Content: {preview}")
    
    # Step 5: Test context retrieval
    print("\n5. Testing context retrieval...")
    query = "What is machine learning?"
    context = vector_store.get_relevant_context(query, k=2)
    print(f"\n📄 Context for '{query}':")
    print("="*50)
    print(context[:500] + "..." if len(context) > 500 else context)
    
    # Step 6: Save vector store
    print("\n6. Saving vector store...")
    save_path = "data/vector_store"
    vector_store.save_vector_store(save_path)
    
    print(f"\n🎉 Vector store test completed successfully!")
    print(f"📝 Ready for Phase 5: Query Processing & LLM Integration")
    
    return vector_store

if __name__ == "__main__":
    test_vector_store()