from langchain_groq import ChatGroq
from vector_store import VectorStore
from typing import Dict, List, Optional, Tuple
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):  # ✅ Updated model
        """
        Initialize query processor with Groq LLM.
        
        LEARNING NOTE: What is RAG (Retrieval-Augmented Generation)?
        - Traditional LLM: Question → LLM → Answer (limited by training data)
        - RAG: Question → Retrieve Context → LLM(Question + Context) → Better Answer
        - This gives us up-to-date, source-attributed, accurate answers
        
        LEARNING NOTE: Model Change
        - Old: mixtral-8x7b-32768 (decommissioned)
        - New: llama-3.3-70b-versatile (recommended replacement)
        - llama-3.3-70b means 70 billion parameters (very capable)
        - "versatile" means good for general tasks like Q&A
        
        Args:
            groq_api_key: Groq API key for LLM access
            model_name: Groq model to use (llama-3.3-70b-versatile is current best)
        """
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.vector_store = None
        
        # LEARNING NOTE: ChatGroq is our interface to Groq's LLMs
        # llama-3.3-70b-versatile means:
        # - llama-3.3: Meta's latest Llama model architecture
        # - 70b: 70 billion parameters (very smart!)
        # - versatile: Optimized for general-purpose tasks
        print(f"🔄 Initializing Groq LLM: {model_name}")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.1,  # Low temperature = more focused, less creative
            max_tokens=1000,  # Limit response length
            timeout=30,       # Timeout after 30 seconds
        )
        print("✅ Groq LLM initialized successfully")
    
    def load_vector_store(self, vector_store: VectorStore) -> None:
        """
        Load vector store for document retrieval.
        
        LEARNING NOTE: This connects our vector search to the LLM
        - Vector store provides relevant context
        - LLM uses that context to generate answers
        - Together they create a RAG system
        """
        self.vector_store = vector_store
        info = vector_store.get_store_info()
        print(f"📚 Vector store loaded: {info['total_chunks']} chunks from {info['unique_sources']} sources")
    
    def create_rag_prompt(self, question: str, context: str) -> str:
        """
        Create a well-structured prompt for RAG.
        
        LEARNING NOTE: Prompt engineering is crucial for good results
        - Clear instructions help the LLM understand its role
        - Context section provides retrieved information
        - Specific formatting requirements ensure consistent output
        - Source attribution prevents hallucination
        
        Args:
            question: User's question
            context: Retrieved context from vector store
            
        Returns:
            Formatted prompt for the LLM
        """
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context
- If the context doesn't contain enough information to answer the question, say so
- Always cite your sources by mentioning the document/source name
- Be concise but comprehensive
- If you're unsure about something, express that uncertainty

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        return prompt
    
    def process_query(self, question: str, k: int = 3, include_sources: bool = True) -> Dict:
        """
        Process a user query through the complete RAG pipeline.
        
        LEARNING NOTE: This is the heart of our RAG system!
        Step by step:
        1. Use vector store to find relevant chunks
        2. Combine chunks into context
        3. Create prompt with question + context
        4. Send to LLM for answer generation
        5. Return answer with source attribution
        
        Args:
            question: User's question
            k: Number of relevant chunks to retrieve
            include_sources: Whether to include source information
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.vector_store:
            raise ValueError("Vector store not loaded. Call load_vector_store() first.")
        
        start_time = time.time()
        
        # Step 1: Retrieve relevant context
        print(f"🔍 Retrieving context for: '{question}'")
        retrieval_start = time.time()
        
        search_results = self.vector_store.similarity_search(question, k=k)
        context = self.vector_store.get_relevant_context(question, k=k)
        
        retrieval_time = time.time() - retrieval_start
        print(f"✅ Retrieved {len(search_results)} relevant chunks ({retrieval_time:.2f}s)")
        
        # Step 2: Create RAG prompt
        prompt = self.create_rag_prompt(question, context)
        
        # Step 3: Generate answer using LLM
        print("🤖 Generating answer with Groq LLM...")
        generation_start = time.time()
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
            generation_time = time.time() - generation_start
            
            print(f"✅ Answer generated ({generation_time:.2f}s)")
            
        except Exception as e:
            generation_time = time.time() - generation_start  # Calculate time even on error
            total_time = time.time() - start_time
            
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "Sorry, I encountered an error while generating the answer.",
                "error": str(e),
                "sources": [],
                "question": question,
                "context_chunks": len(search_results),
                "processing_time": {
                    "retrieval": round(retrieval_time, 2),
                    "generation": round(generation_time, 2),
                    "total": round(total_time, 2)
                }
            }
        
        # Step 4: Extract source information
        sources = []
        if include_sources and search_results:
            for content, metadata, score in search_results:
                sources.append({
                    "title": metadata.get('title', 'Unknown Source'),
                    "chunk_id": metadata.get('chunk_id', 0),
                    "chunk_count": metadata.get('chunk_count', 1),
                    "similarity_score": round(score, 3),
                    "content_preview": content[:150] + "..." if len(content) > 150 else content
                })
        
        total_time = time.time() - start_time
        
        # Return complete response
        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "context_chunks": len(search_results),
            "processing_time": {
                "retrieval": round(retrieval_time, 2),
                "generation": round(generation_time, 2),
                "total": round(total_time, 2)
            }
        }
    
    def batch_process_queries(self, questions: List[str], k: int = 3) -> List[Dict]:
        """
        Process multiple queries in batch.
        
        LEARNING NOTE: Useful for testing and evaluation
        - Process multiple questions at once
        - Compare performance across different query types
        - Useful for building evaluation datasets
        """
        results = []
        
        print(f"🔄 Processing {len(questions)} questions in batch...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i}/{len(questions)} ---")
            result = self.process_query(question, k=k)
            results.append(result)
        
        print(f"\n✅ Batch processing completed!")
        return results
    
    def interactive_chat(self) -> None:
        """
        Start an interactive chat session.
        
        LEARNING NOTE: This creates a simple chat interface
        - Users can ask questions interactively
        - Each question goes through the full RAG pipeline
        - Shows sources and processing time
        - Useful for testing and demos
        """
        if not self.vector_store:
            print("❌ Vector store not loaded. Please load vector store first.")
            return
        
        print("\n🤖 Interactive Q&A Chat Started!")
        print("Type 'quit' to exit, 'help' for commands")
        print("="*50)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                elif question.lower() == 'help':
                    print("\nCommands:")
                    print("  help - Show this help")
                    print("  quit - Exit chat")
                    print("  Just type your question normally!")
                    continue
                    
                elif not question:
                    print("Please enter a question!")
                    continue
                
                # Process the question
                result = self.process_query(question)
                
                # Display results
                print(f"\n🤖 **Answer:**")
                print(result['answer'])
                
                if result['sources']:
                    print(f"\n📚 **Sources:**")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"   {i}. {source['title']} (Chunk {source['chunk_id']+1}/{source['chunk_count']})")
                        print(f"      Similarity: {source['similarity_score']}")
                
                print(f"\n⏱️  Processing time: {result['processing_time']['total']}s")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

# Test function
def test_query_processor():
    """
    Test the complete RAG pipeline.
    
    LEARNING NOTE: This demonstrates the full system working together:
    1. Load vector store (from Phase 4)
    2. Initialize LLM connection
    3. Process various types of questions
    4. Show how context retrieval + generation works
    """

    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Load API key
    from config import GROQ_API_KEY
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in config.py")
        return
    
    print("🧪 Testing Complete RAG Pipeline...")
    
    # Step 1: Load existing vector store
    print("\n1. Loading vector store...")
    vector_store = VectorStore()
    
    # Try to load saved vector store, create if doesn't exist
    try:
        vector_store.load_vector_store("data/vector_store")
        print("✅ Loaded existing vector store")
    except FileNotFoundError:
        print("🔄 Creating new vector store...")
        from src.document_processor import DocumentProcessor
        from src.document_loader import PDFDocumentLoader
        
        loader = PDFDocumentLoader()
        documents = loader.create_sample_documents()
        processor = DocumentProcessor()
        chunks = processor.process_documents(documents)
        vector_store.create_vector_store(chunks)
        vector_store.save_vector_store("data/vector_store")
    
    # Step 2: Initialize query processor
    print("\n2. Initializing query processor...")
    query_processor = QueryProcessor(GROQ_API_KEY)
    query_processor.load_vector_store(vector_store)
    
    # Step 3: Test different types of queries
    print("\n3. Testing various query types...")
    
    test_questions = [
        "What is machine learning?",
        "How do neural networks work?", 
        "What is the difference between AI and machine learning?",
        "Can you explain backpropagation?",
        "What are the applications of artificial intelligence?"
    ]
    
    for question in test_questions:
        print(f"\n" + "="*60)
        print(f"🔍 Question: {question}")
        
        result = query_processor.process_query(question, k=2)
        
        print(f"\n🤖 Answer:")
        print(result['answer'])
        
        # ✅ Fixed: Better error handling for sources and timing
        if 'error' in result:
            print(f"⚠️  Error occurred: {result['error']}")
        
        sources = result.get('sources', [])
        print(f"\n📚 Sources ({len(sources)}):")
        for i, source in enumerate(sources, 1):
            print(f"   {i}. {source['title']} (Score: {source['similarity_score']})")
            print(f"      Preview: {source['content_preview']}")
        
        # ✅ Fixed: Safe access to processing_time
        if 'processing_time' in result:
            timing = result['processing_time']
            print(f"\n⏱️  Timing: Retrieval {timing['retrieval']}s, "
                  f"Generation {timing['generation']}s, Total {timing['total']}s")
        
        # Add a small delay between questions to avoid rate limiting
        time.sleep(1)
    
    print(f"\n🎉 RAG Pipeline test completed successfully!")
    print(f"🚀 Ready for Phase 6: Complete Application & UI!")
    
    return query_processor

if __name__ == "__main__":
    test_query_processor()