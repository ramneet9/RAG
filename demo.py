"""
Demo script for RAG Application

This script demonstrates the RAG system with interactive conversation.
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
from config import PDF_URLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_rag_system():
    """Setup the complete RAG system."""
    print("🔧 Setting up RAG system...")
    
    # Initialize components
    pdf_processor = PDFProcessor()
    text_chunker = TextChunker()
    vector_store = VectorStore()
    llm_client = HybridLLMClient()
    conversation_manager = ConversationManager(llm_client, vector_store)
    
    # Check if vector database already exists
    if vector_store.load_index():
        print("✅ Loaded existing vector database")
        return conversation_manager
    
    # Process PDFs if database doesn't exist
    print("📄 Processing PDFs...")
    pdf_processor.download_pdfs(PDF_URLS)
    texts = pdf_processor.extract_texts()
    
    if not texts:
        raise ValueError("No texts extracted from PDFs")
    
    print("✂️ Chunking texts...")
    chunks = text_chunker.chunk_texts(texts)
    
    if not chunks:
        raise ValueError("No chunks created from texts")
    
    print("🔍 Creating vector database...")
    vector_store.create_index(chunks)
    
    print("✅ RAG system setup complete!")
    return conversation_manager

def interactive_demo():
    """Run interactive demo."""
    print("🤖 RAG Application Interactive Demo")
    print("=" * 50)
    print("Ask questions about the research papers!")
    print("Type 'quit' to exit, 'clear' to clear memory, 'history' to see conversation history")
    print("=" * 50)
    
    try:
        conversation_manager = setup_rag_system()
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                conversation_manager.clear_history()
                print("🧹 Conversation memory cleared!")
                continue
            elif user_input.lower() == 'history':
                history = conversation_manager.get_conversation_history()
                if history:
                    print("\n📜 Conversation History:")
                    for i, turn in enumerate(history, 1):
                        print(f"  {i}. Q: {turn['query']}")
                        print(f"     A: {turn['response'][:100]}...")
                else:
                    print("📜 No conversation history yet.")
                continue
            elif not user_input:
                continue
            
            print("🤖 Thinking...")
            
            try:
                result = conversation_manager.generate_response(user_input)
                
                print(f"\n🤖 Assistant: {result['response']}")
                print(f"📊 Context used: {len(result['context'])} characters")
                print(f"💭 Memory turns: {result['conversation_turns']}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                logger.error(f"Error in interactive demo: {str(e)}")
    
    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        logger.error(f"Setup failed: {str(e)}")

def quick_demo():
    """Run quick demo with predefined questions."""
    print("🚀 RAG Application Quick Demo")
    print("=" * 40)
    
    try:
        conversation_manager = setup_rag_system()
        
        demo_questions = [
            "What is the Transformer architecture?",
            "How does BERT differ from GPT?",
            "What are attention mechanisms?",
            "What are the limitations of these models?"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            print("🤖 Thinking...")
            
            result = conversation_manager.generate_response(question)
            
            print(f"🤖 Answer: {result['response']}")
            print(f"📊 Context: {len(result['context'])} characters")
            print(f"💭 Memory: {result['conversation_turns']} turns")
            print("-" * 40)
    
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.error(f"Demo failed: {str(e)}")

def main():
    """Main demo function."""
    print("🎯 RAG Application Demo")
    print("Choose demo mode:")
    print("1. Interactive demo (ask your own questions)")
    print("2. Quick demo (predefined questions)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        interactive_demo()
    elif choice == "2":
        quick_demo()
    else:
        print("Invalid choice. Running quick demo...")
        quick_demo()

if __name__ == "__main__":
    main()
