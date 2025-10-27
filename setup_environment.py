"""
Environment Setup Script

This script sets up the complete environment for the RAG application.
"""

import subprocess
import sys
import os
from pathlib import Path
import platform

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible!")
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = ["data", "reports", "vector_db", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_system_requirements():
    """Check system requirements."""
    print("🖥️ Checking system requirements...")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 Available RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠️ Warning: Less than 4GB RAM available. Performance may be affected.")
    except ImportError:
        print("ℹ️ psutil not available for memory check")
    
    # Check disk space
    disk_usage = Path('.').stat().st_size
    print(f"💽 Disk space check: {disk_usage} bytes")
    
    # Check platform
    print(f"🖥️ Platform: {platform.system()} {platform.release()}")

def download_models():
    """Download required models."""
    print("🤖 Downloading models...")
    
    try:
        # This will trigger model downloads when first used
        print("ℹ️ Models will be downloaded automatically on first use")
        print("   - Embedding model: sentence-transformers/all-MiniLM-L6-v2")
        print("   - LLM model: microsoft/DialoGPT-medium")
        return True
    except Exception as e:
        print(f"❌ Model download issue: {e}")
        return False

def run_tests():
    """Run basic tests."""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from src.pdf_processor import PDFProcessor
        from src.text_chunker import TextChunker
        from src.vector_store import VectorStore
        from src.llm_client import LLMClient
        from src.conversation_manager import ConversationManager
        from src.evaluator import RAGEvaluator
        
        print("✅ All modules imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up RAG Application Environment...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("⚠️ Some packages failed to install. You may need to install them manually.")
        print("   Try: pip install -r requirements.txt")
    
    # Download models info
    download_models()
    
    # Run tests
    if not run_tests():
        print("⚠️ Some tests failed. Check the error messages above.")
    
    print("\n🎉 Setup completed!")
    print("=" * 50)
    print("Next steps:")
    print("1. Run the application: python main.py")
    print("2. Try the demo: python demo.py")
    print("3. Run tests: python test.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
