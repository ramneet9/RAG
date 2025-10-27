"""
Perplexity API Validation and Testing Script

This script validates and tests Perplexity API configuration for the RAG application.
"""

import os
import sys
from pathlib import Path

def check_perplexity_key():
    """Check if Perplexity API key is configured."""
    print("Checking Perplexity API key configuration...")
    
    config_path = Path("config.py")
    if not config_path.exists():
        print("ERROR: config.py not found!")
        return False
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Check for empty API key
    if 'PERPLEXITY_API_KEY = ""' in content:
        print("WARNING: Perplexity API key not configured")
        print("Please add your API key to config.py:")
        print('PERPLEXITY_API_KEY = "your_api_key_here"')
        return False
    else:
        print("SUCCESS: Perplexity API key appears to be configured!")
        return True

def test_perplexity_connection():
    """Test Perplexity API connection."""
    print("Testing Perplexity API connection...")
    
    try:
        from src.hybrid_llm_client import HybridLLMClient
        
        client = HybridLLMClient()
        
        if client.test_connection():
            print("SUCCESS: Perplexity API connection successful!")
            return True
        else:
            print("ERROR: Perplexity API connection failed")
            print("This could be due to:")
            print("- Invalid API key")
            print("- Network connectivity issues")
            print("- Perplexity API service issues")
            print("- Rate limiting")
            return False
            
    except Exception as e:
        print(f"ERROR: Connection test error: {str(e)}")
        print("This could be due to:")
        print("- Missing API key")
        print("- Invalid API key format")
        print("- Network connectivity issues")
        print("- Missing dependencies")
        return False

def test_embedding_model():
    """Test sentence-transformers embedding model."""
    print("Testing sentence-transformers embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDER_MODEL
        
        print(f"Loading model: {EMBEDDER_MODEL}")
        model = SentenceTransformer(EMBEDDER_MODEL)
        
        # Test embedding generation
        test_texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(test_texts)
        
        print(f"SUCCESS: Embedding model loaded successfully!")
        print(f"SUCCESS: Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        return True
        
    except Exception as e:
        print(f"ERROR: Embedding model test error: {str(e)}")
        return False

def test_full_system():
    """Test the complete RAG system."""
    print("Testing complete RAG system...")
    
    try:
        from src.hybrid_llm_client import HybridLLMClient
        from src.vector_store import VectorStore
        
        # Test LLM client
        print("Testing LLM client...")
        llm_client = HybridLLMClient()
        
        # Test vector store
        print("Testing vector store...")
        vector_store = VectorStore()
        
        # Test embedding generation
        print("Testing embedding generation...")
        test_texts = ["Test document content for embedding."]
        embeddings = llm_client.generate_embeddings(test_texts)
        
        print("SUCCESS: Complete RAG system test successful!")
        return True
        
    except Exception as e:
        print(f"ERROR: Full system test error: {str(e)}")
        return False

def show_configuration():
    """Show current configuration."""
    print("Current Configuration:")
    print("=" * 30)
    
    try:
        from config import (
            API_PROVIDER, PERPLEXITY_MODEL, PERPLEXITY_API_BASE,
            EMBEDDER_PROVIDER, EMBEDDER_MODEL,
            CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL
        )
        
        print(f"API Provider: {API_PROVIDER}")
        print(f"Perplexity Model: {PERPLEXITY_MODEL}")
        print(f"API Base URL: {PERPLEXITY_API_BASE}")
        print(f"Embedding Provider: {EMBEDDER_PROVIDER}")
        print(f"Embedding Model: {EMBEDDER_MODEL}")
        print(f"Chunk Size: {CHUNK_SIZE}")
        print(f"Chunk Overlap: {CHUNK_OVERLAP}")
        print(f"Top K Retrieval: {TOP_K_RETRIEVAL}")
        
    except Exception as e:
        print(f"ERROR: Error reading configuration: {str(e)}")

def main():
    """Main validation and testing function."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "check":
            check_perplexity_key()
        elif command == "test":
            test_perplexity_connection()
        elif command == "embedding":
            test_embedding_model()
        elif command == "full":
            test_full_system()
        elif command == "config":
            show_configuration()
        else:
            print("ERROR: Unknown command. Available commands:")
            print("  check     - Check API key configuration")
            print("  test      - Test Perplexity API connection")
            print("  embedding - Test embedding model")
            print("  full      - Test complete RAG system")
            print("  config    - Show current configuration")
    else:
        # Run all tests by default
        print("Perplexity API Validation and Testing")
        print("=" * 40)
        
        print("\n1. Checking configuration...")
        config_ok = check_perplexity_key()
        
        if config_ok:
            print("\n2. Testing API connection...")
            api_ok = test_perplexity_connection()
            
            print("\n3. Testing embedding model...")
            embedding_ok = test_embedding_model()
            
            if api_ok and embedding_ok:
                print("\n4. Testing complete system...")
                system_ok = test_full_system()
                
                if system_ok:
                    print("\nSUCCESS: All tests passed! Your RAG system is ready to use.")
                    print("Run: python main.py")
                else:
                    print("\nWARNING: System test failed. Check the errors above.")
            else:
                print("\nWARNING: Some tests failed. Fix the issues before running the system.")
        else:
            print("\nERROR: Configuration check failed. Please configure your API key first.")

if __name__ == "__main__":
    main()
