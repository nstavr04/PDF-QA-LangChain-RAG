import os
import fitz  # PyMuPDF
import PyPDF2
from typing import Dict, List, Optional
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFDocumentLoader:
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt']
    
    def load_from_pdf_bytes(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> Dict:
        """Load document from PDF bytes (for Streamlit file upload)."""
        try:
            # Try PyMuPDF first (better text extraction)
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_content = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text_content += page.get_text() + "\n"
            
            doc.close()
            
            if not text_content.strip():
                raise ValueError("No text content found in PDF")
            
            metadata = {
                "source": "pdf_upload",
                "title": filename,
                "length": len(text_content),
                "type": "pdf",
                "pages": doc.page_count if 'doc' in locals() else 0
            }
            
            logger.info(f"Loaded PDF: {filename} ({len(text_content)} characters, {metadata['pages']} pages)")
            return {
                "content": text_content.strip(),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"PyMuPDF failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                pdf_file = io.BytesIO(pdf_bytes)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                
                if not text_content.strip():
                    raise ValueError("No text content found in PDF")
                
                metadata = {
                    "source": "pdf_upload",
                    "title": filename,
                    "length": len(text_content),
                    "type": "pdf",
                    "pages": len(pdf_reader.pages)
                }
                
                logger.info(f"Loaded PDF with PyPDF2: {filename} ({len(text_content)} characters)")
                return {
                    "content": text_content.strip(),
                    "metadata": metadata
                }
                
            except Exception as e2:
                logger.error(f"Both PDF readers failed: {e2}")
                raise ValueError(f"Could not extract text from PDF: {e2}")
    
    def load_from_pdf_file(self, file_path: str) -> Dict:
        """Load document from PDF file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()
        
        return self.load_from_pdf_bytes(pdf_bytes, os.path.basename(file_path))
    
    def load_from_text(self, text_content: str, title: str = "Direct Input") -> Dict:
        """Load document from direct text input."""
        if not text_content.strip():
            raise ValueError("Text content cannot be empty")
        
        metadata = {
            "source": "direct_input",
            "title": title,
            "length": len(text_content),
            "type": "text"
        }
        
        logger.info(f"Loaded text: {title} ({len(text_content)} characters)")
        return {
            "content": text_content.strip(),
            "metadata": metadata
        }
    
    def create_sample_documents(self) -> List[Dict]:
        """Create sample educational documents for testing."""
        samples = [
            {
                "title": "Machine Learning Fundamentals",
                "content": """
# Machine Learning Fundamentals

## Introduction
Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

## Types of Machine Learning

### Supervised Learning
Supervised learning algorithms build a mathematical model of training data, known as training data, that contains both the inputs and the desired outputs. The goal is to learn a general rule that maps inputs to outputs.

Examples:
- Classification: Email spam detection, image recognition
- Regression: Price prediction, weather forecasting

### Unsupervised Learning
Unsupervised learning algorithms take a set of data that contains only inputs, and find structure in the data, like grouping or clustering of data points.

Examples:
- Clustering: Customer segmentation, gene sequencing
- Association: Market basket analysis
- Dimensionality reduction: Data visualization, feature selection

### Reinforcement Learning
Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.

Examples:
- Game playing: Chess, Go, video games
- Robotics: Robot navigation, manipulation
- Finance: Trading strategies, portfolio management

## Machine Learning Pipeline

1. **Data Collection**: Gathering relevant data from various sources
2. **Data Preprocessing**: Cleaning and preparing data for analysis
3. **Feature Engineering**: Selecting and transforming variables
4. **Model Selection**: Choosing appropriate algorithms
5. **Training**: Teaching the model using training data
6. **Evaluation**: Testing model performance
7. **Deployment**: Implementing the model in production
8. **Monitoring**: Tracking model performance over time

## Key Concepts

### Overfitting and Underfitting
- **Overfitting**: Model performs well on training data but poorly on new data
- **Underfitting**: Model is too simple to capture underlying patterns
- **Solution**: Cross-validation, regularization, proper model complexity

### Bias-Variance Tradeoff
- **Bias**: Error from overly simplistic assumptions
- **Variance**: Error from sensitivity to small fluctuations
- **Goal**: Find optimal balance between bias and variance

### Cross-Validation
A technique for assessing how well a model will generalize to an independent dataset. Common methods include:
- K-fold cross-validation
- Leave-one-out cross-validation
- Stratified cross-validation

## Conclusion
Machine learning is a powerful tool for extracting insights from data and making predictions. Success requires careful attention to data quality, appropriate algorithm selection, and rigorous evaluation methods.
                """.strip()
            }
        ]
        
        documents = []
        for sample in samples:
            doc = self.load_from_text(sample["content"], sample["title"])
            documents.append(doc)
        
        return documents

# Test function
def test_pdf_loader():
    """Test the PDF document loader."""
    loader = PDFDocumentLoader()
    
    # Test sample documents
    samples = loader.create_sample_documents()
    print(f"✅ Created {len(samples)} sample documents")
    for doc in samples:
        print(f"- {doc['metadata']['title']}: {doc['metadata']['length']} chars")
        print(f"  Preview: {doc['content'][:200]}...")

if __name__ == "__main__":
    test_pdf_loader()