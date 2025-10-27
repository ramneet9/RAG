"""
Test script for RAG Application

This script tests individual components and the overall system.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pdf_processor import PDFProcessor
from src.text_chunker import TextChunker
from src.vector_store import VectorStore
from src.hybrid_llm_client import HybridLLMClient
from src.conversation_manager import ConversationManager
from src.evaluator import RAGEvaluator
from config import PDF_URLS, EVALUATION_QUESTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pdf_processor():
    """Test PDF processing functionality."""
    print("🧪 Testing PDF Processor...")
    
    processor = PDFProcessor()
    
    # Test with a single PDF URL
    test_urls = [PDF_URLS[0]]  # Just test with first PDF
    downloaded_files = processor.download_pdfs(test_urls)
    
    if downloaded_files:
        print(f"✅ Downloaded {len(downloaded_files)} PDF(s)")
        
        # Test text extraction
        texts = processor.extract_texts()
        if texts:
            print(f"✅ Extracted text from {len(texts)} PDF(s)")
            print(f"   Sample text length: {len(texts[0]['text'])} characters")
            return texts
        else:
            print("❌ Failed to extract text")
            return None
    else:
        print("❌ Failed to download PDFs")
        return None

def test_text_chunker(texts):
    """Test text chunking functionality."""
    print("🧪 Testing Text Chunker...")
    
    if not texts:
        print("❌ No texts to chunk")
        return None
    
    chunker = TextChunker()
    chunks = chunker.chunk_texts(texts)
    
    if chunks:
        print(f"✅ Created {len(chunks)} chunks")
        print(f"   Sample chunk length: {len(chunks[0]['text'])} characters")
        return chunks
    else:
        print("❌ Failed to create chunks")
        return None

def test_vector_store(chunks):
    """Test vector store functionality."""
    print("🧪 Testing Vector Store...")
    
    if not chunks:
        print("❌ No chunks to process")
        return None
    
    vector_store = VectorStore()
    
    try:
        vector_store.create_index(chunks)
        print("✅ Vector index created successfully")
        
        # Test search
        test_query = "transformer architecture"
        results = vector_store.search(test_query, k=3)
        
        if results:
            print(f"✅ Search returned {len(results)} results")
            print(f"   Top result score: {results[0]['score']:.3f}")
            return vector_store
        else:
            print("❌ Search returned no results")
            return None
            
    except Exception as e:
        print(f"❌ Vector store error: {str(e)}")
        return None

def test_llm_client():
    """Test LLM client functionality."""
    print("🧪 Testing LLM Client...")
    
    try:
        llm_client = HybridLLMClient()
        print("✅ LLM client initialized successfully")
        
        # Test response generation
        test_context = "The Transformer architecture uses attention mechanisms."
        test_query = "What is the Transformer architecture?"
        
        response = llm_client.generate_response(test_context, test_query)
        
        if response and len(response) > 10:
            print(f"✅ Generated response: {response[:100]}...")
            return llm_client
        else:
            print("❌ Failed to generate meaningful response")
            return None
            
    except Exception as e:
        print(f"❌ LLM client error: {str(e)}")
        return None

def test_conversation_manager(llm_client, vector_store):
    """Test conversation manager functionality."""
    print("🧪 Testing Conversation Manager...")
    
    if not llm_client or not vector_store:
        print("❌ Missing dependencies")
        return None
    
    try:
        conversation_manager = ConversationManager(llm_client, vector_store)
        
        # Test response generation
        test_query = "What is attention mechanism?"
        result = conversation_manager.generate_response(test_query)
        
        if result and "response" in result:
            print(f"✅ Generated response: {result['response'][:100]}...")
            print(f"   Conversation turns: {result['conversation_turns']}")
            return conversation_manager
        else:
            print("❌ Failed to generate response")
            return None
            
    except Exception as e:
        print(f"❌ Conversation manager error: {str(e)}")
        return None

def test_evaluator(conversation_manager):
    """Test evaluator functionality."""
    print("🧪 Testing Evaluator...")
    
    if not conversation_manager:
        print("❌ Missing conversation manager")
        return None
    
    try:
        evaluator = RAGEvaluator(conversation_manager)
        
        # Test with a single question
        test_question = "What is the main contribution of the Transformer architecture?"
        result = evaluator.evaluate_single_question(test_question)
        
        if result and "metrics" in result:
            print("✅ Evaluation completed successfully")
            print(f"   Overall score: {result['metrics']['overall_score']:.3f}")
            print(f"   Relevance: {result['metrics']['relevance']:.3f}")
            print(f"   Accuracy: {result['metrics']['accuracy']:.3f}")
            return evaluator
        else:
            print("❌ Evaluation failed")
            return None
            
    except Exception as e:
        print(f"❌ Evaluator error: {str(e)}")
        return None

def main():
    """Run all tests."""
    print("🚀 Starting RAG Application Tests...")
    
    # Test individual components
    texts = test_pdf_processor()
    chunks = test_text_chunker(texts)
    vector_store = test_vector_store(chunks)
    llm_client = test_llm_client()
    conversation_manager = test_conversation_manager(llm_client, vector_store)
    evaluator = test_evaluator(conversation_manager)
    
    # Summary
    print("\n📊 Test Summary:")
    components = [
        ("PDF Processor", texts is not None),
        ("Text Chunker", chunks is not None),
        ("Vector Store", vector_store is not None),
        ("LLM Client", llm_client is not None),
        ("Conversation Manager", conversation_manager is not None),
        ("Evaluator", evaluator is not None)
    ]
    
    passed = sum(1 for _, status in components if status)
    total = len(components)
    
    for component, status in components:
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {component}")
    
    print(f"\n🎯 Overall: {passed}/{total} components passed")
    
    if passed == total:
        print("🎉 All tests passed! The RAG application is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
