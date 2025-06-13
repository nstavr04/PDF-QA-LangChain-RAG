from src.document_loader import PDFDocumentLoader
from src.document_processor import DocumentProcessor

def test_document_processing():
    """Test the complete document processing pipeline."""
    print("🧪 Testing Document Processing Pipeline...")
    
    # Step 1: Load documents
    print("\n1. Loading documents...")
    loader = PDFDocumentLoader()
    
    # Test with sample documents
    documents = loader.create_sample_documents()
    print(f"✅ Loaded {len(documents)} documents")
    
    # Add the sample lecture text file
    try:
        with open("data/sample_pdfs/sample_lecture.txt", "r") as f:
            lecture_content = f.read()
        
        lecture_doc = loader.load_from_text(lecture_content, "Neural Networks Lecture")
        documents.append(lecture_doc)
        print(f"✅ Added sample lecture file")
    except FileNotFoundError:
        print("⚠️  Sample lecture file not found, continuing with generated samples")
    
    # Step 2: Process documents into chunks
    print("\n2. Processing documents into chunks...")
    processor = DocumentProcessor()
    chunks = processor.process_documents(documents)
    
    # Step 3: Show statistics
    print("\n3. Processing Results:")
    stats = processor.get_processing_stats(documents, chunks)
    for key, value in stats.items():
        print(f"   📈 {key.replace('_', ' ').title()}: {value}")
    
    # Step 4: Show chunk preview
    processor.get_chunk_preview(chunks, max_chunks=2)
    
    # Step 5: Test chunk quality
    print("\n4. Testing chunk quality...")
    
    # Test for empty chunks
    empty_chunks = [c for c in chunks if not c['content'].strip()]
    print(f"   Empty chunks: {len(empty_chunks)}")
    
    # Test chunk sizes
    chunk_sizes = [len(c['content']) for c in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0
    min_size = min(chunk_sizes) if chunk_sizes else 0
    
    print(f"   Chunk size - Avg: {avg_size:.0f}, Max: {max_size}, Min: {min_size}")
    
    # Test metadata preservation
    chunks_with_metadata = [c for c in chunks if c.get('metadata', {}).get('title')]
    print(f"   Chunks with metadata: {len(chunks_with_metadata)}/{len(chunks)}")
    
    print(f"\n🎉 Document processing test completed successfully!")
    print(f"📝 Ready for Phase 4: Vector Store Implementation")
    
    return chunks

if __name__ == "__main__":
    test_document_processing()