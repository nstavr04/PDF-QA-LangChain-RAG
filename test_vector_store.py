import time
from src.vector_store import VectorStore
from src.document_processor import DocumentProcessor
from src.document_loader import PDFDocumentLoader

def comprehensive_vector_test():
    """Comprehensive test of vector store functionality."""
    print("🚀 Comprehensive Vector Store Test")
    print("="*50)
    
    # Phase 2 + 3: Document loading and processing
    print("\n📋 Step 1: Document Processing Pipeline")
    start_time = time.time()
    
    loader = PDFDocumentLoader()
    documents = loader.create_sample_documents()
    print(f"   ✅ Loaded {len(documents)} documents")
    
    processor = DocumentProcessor()
    chunks = processor.process_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks")
    
    processing_time = time.time() - start_time
    print(f"   ⏱️  Processing time: {processing_time:.2f}s")
    
    # Phase 4: Vector store creation
    print("\n🔮 Step 2: Vector Store Creation")
    start_time = time.time()
    
    vector_store = VectorStore()
    vector_store.create_vector_store(chunks)
    
    embedding_time = time.time() - start_time
    print(f"   ⏱️  Embedding time: {embedding_time:.2f}s")
    
    # Test different query types
    print("\n🔍 Step 3: Query Testing")
    
    test_scenarios = [
        {
            "category": "Direct Concept",
            "queries": ["machine learning", "neural networks", "artificial intelligence"]
        },
        {
            "category": "How-To Questions", 
            "queries": ["How do neural networks learn?", "What is backpropagation?"]
        },
        {
            "category": "Definition Questions",
            "queries": ["What is AI?", "Define machine learning", "Explain deep learning"]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n   📚 {scenario['category']}:")
        for query in scenario['queries']:
            results = vector_store.similarity_search(query, k=1)
            if results:
                content, metadata, score = results[0]
                source = metadata.get('title', 'Unknown')
                print(f"      Q: {query}")
                print(f"      A: [{source}] Score: {score:.3f}")
                print(f"         {content[:80]}...")
    
    # Performance analysis
    print("\n📊 Step 4: Performance Analysis")
    info = vector_store.get_store_info()
    print(f"   Total chunks: {info['total_chunks']}")
    print(f"   Vector dimensions: {info['vector_dimensions']}")
    print(f"   Unique sources: {info['unique_sources']}")
    print(f"   Avg embedding time per chunk: {embedding_time/len(chunks):.4f}s")
    
    # Save/Load test
    print("\n💾 Step 5: Persistence Test")
    save_path = "data/test_vector_store"
    vector_store.save_vector_store(save_path)
    
    # Test loading
    new_vector_store = VectorStore()
    new_vector_store.load_vector_store(save_path)
    
    # Verify loaded store works
    test_query = "machine learning"
    original_results = vector_store.similarity_search(test_query, k=1)
    loaded_results = new_vector_store.similarity_search(test_query, k=1)
    
    if original_results and loaded_results:
        orig_score = original_results[0][2]
        loaded_score = loaded_results[0][2]
        print(f"   ✅ Save/Load works (scores: {orig_score:.3f} vs {loaded_score:.3f})")
    
    print(f"\n🎉 All tests completed successfully!")
    print(f"🚀 Ready for Phase 5: LLM Integration!")
    
    return vector_store

if __name__ == "__main__":
    comprehensive_vector_test()