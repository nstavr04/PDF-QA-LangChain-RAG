from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, List
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize document processor with chunking parameters.
        
        LEARNING NOTE: 
        - chunk_size: Maximum characters per chunk (e.g., 1000)
        - chunk_overlap: Characters that overlap between chunks (e.g., 200)
        - Overlap helps maintain context between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter with smart separators
        # LEARNING NOTE: RecursiveCharacterTextSplitter tries to split on:
        # 1. Paragraphs first (\n\n) - preserves document structure
        # 2. Then lines (\n) - keeps sentences together
        # 3. Then sentences (.) - keeps words together  
        # 4. Then words ( ) - keeps characters together
        # 5. Finally characters - as last resort
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",  # Paragraph breaks (best)
                "\n",    # Line breaks
                ".",     # Sentences
                " ",     # Words
                ""       # Characters (worst case)
            ],
            keep_separator=True,    # Keep the separator character
            is_separator_regex=False  # Use literal string matching
        )
    
    def process_document(self, document: Dict) -> List[Dict]:
        """
        Process a single document into chunks.
        
        LEARNING NOTE: This is the core chunking logic
        - Takes one document and splits it into multiple chunks
        - Preserves metadata (source info) in each chunk
        - Adds chunk-specific metadata for tracking
        
        Args:
            document: Document dict with 'content' and 'metadata' keys
            
        Returns:
            List of chunk documents with preserved metadata
        """
        try:
            content = document.get('content', '')
            metadata = document.get('metadata', {})
            
            if not content.strip():
                logger.warning("Empty document content provided")
                return []
            
            # LEARNING NOTE: This is where the magic happens!
            # split_text() uses our smart separators to break text intelligently
            chunks = self.text_splitter.split_text(content)
            
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    # LEARNING NOTE: We copy original metadata and add chunk info
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_id': i,                    # Which chunk number (0, 1, 2...)
                        'chunk_count': len(chunks),       # Total chunks from this document
                        'chunk_size': len(chunk),         # Size of this specific chunk
                        'original_length': len(content)   # Size of original document
                    })
                    
                    processed_chunks.append({
                        'content': chunk.strip(),
                        'metadata': chunk_metadata
                    })
            
            logger.info(f"Processed document '{metadata.get('title', 'Unknown')}' into {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process multiple documents into chunks.
        
        LEARNING NOTE: This handles batch processing
        - Takes a list of documents
        - Processes each one individually
        - Combines all chunks into one big list
        - Continues even if one document fails
        
        Args:
            documents: List of document dicts
            
        Returns:
            List of all chunk documents from all sources
        """
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self.process_document(doc)
                all_chunks.extend(chunks)  # Add chunks to master list
            except Exception as e:
                logger.error(f"Failed to process document: {e}")
                continue  # Skip failed documents, keep processing others
        
        logger.info(f"Total processed chunks: {len(all_chunks)}")
        return all_chunks
    
    def get_chunk_preview(self, chunks: List[Dict], max_chunks: int = 3) -> None:
        """
        Print preview of chunks for debugging.
        
        LEARNING NOTE: This helps us see what our chunking produced
        - Shows first few chunks
        - Displays metadata and content preview
        - Useful for debugging chunk quality
        """
        print(f"\n📄 Document Chunks Preview (showing {min(max_chunks, len(chunks))} of {len(chunks)}):")
        
        for i, chunk in enumerate(chunks[:max_chunks]):
            metadata = chunk['metadata']
            content_preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {metadata.get('title', 'Unknown')}")
            print(f"Chunk {metadata.get('chunk_id', 0)+1}/{metadata.get('chunk_count', 1)}")
            print(f"Size: {metadata.get('chunk_size', 0)} chars")
            print(f"Content: {content_preview}")
    
    def get_processing_stats(self, original_docs: List[Dict], processed_chunks: List[Dict]) -> Dict:
        """
        Get processing statistics.
        
        LEARNING NOTE: This gives us insights into our chunking effectiveness
        - How many documents became how many chunks?
        - Are chunks the right size?
        - Did we lose or gain text in processing?
        """
        total_original_chars = sum(len(doc.get('content', '')) for doc in original_docs)
        total_chunk_chars = sum(len(chunk.get('content', '')) for chunk in processed_chunks)
        
        return {
            'original_documents': len(original_docs),
            'total_chunks': len(processed_chunks),
            'original_characters': total_original_chars,
            'processed_characters': total_chunk_chars,
            'average_chunk_size': total_chunk_chars // len(processed_chunks) if processed_chunks else 0,
            'chunk_size_setting': self.chunk_size,
            'chunk_overlap_setting': self.chunk_overlap
        }

# Test function
def test_document_processor():
    """
    Test the document processor.
    
    LEARNING NOTE: This demonstrates the full pipeline:
    1. Load documents (from Phase 2)
    2. Process them into chunks (Phase 3)
    3. Show statistics and preview
    """
    from src.document_loader import PDFDocumentLoader
    
    print("🔄 Testing Document Processor...")
    
    # Load sample documents
    loader = PDFDocumentLoader()
    documents = loader.create_sample_documents()
    
    # Process documents
    processor = DocumentProcessor()
    chunks = processor.process_documents(documents)
    
    # Show results
    stats = processor.get_processing_stats(documents, chunks)
    print(f"\n📊 Processing Statistics:")
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Show preview
    processor.get_chunk_preview(chunks)
    
    print(f"\n✅ Document processing test completed!")
    return chunks

if __name__ == "__main__":
    test_document_processor()