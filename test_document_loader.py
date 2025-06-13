from src.document_loader import PDFDocumentLoader

def test_all_functionality():
    """Test the PDF document loader functionality."""
    print("🧪 Testing PDF Document Q&A System...")
    
    loader = PDFDocumentLoader()
    
    # Test 1: Sample documents
    print("\n1. Testing sample document creation...")
    samples = loader.create_sample_documents()
    print(f"✅ Created {len(samples)} sample documents")
    
    for doc in samples:
        print(f"   - {doc['metadata']['title']}: {doc['metadata']['length']} chars")
    
    # Test 2: Direct text input
    print("\n2. Testing direct text input...")
    test_text = "This is a test document about artificial intelligence and machine learning concepts."
    doc = loader.load_from_text(test_text, "Test Document")
    print(f"✅ Text input successful: {doc['metadata']['length']} characters")
    
    print("\n🎉 All tests passed! Ready for Phase 3.")

if __name__ == "__main__":
    test_all_functionality()