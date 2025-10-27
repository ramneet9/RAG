# Free RAG Stack Setup Guide

## Overview

This guide helps you set up a completely free RAG (Retrieval-Augmented Generation) application using the best open-source tools:

- **LLM**: Llama3, Mistral, or Phi-3-mini via Ollama
- **Embedder**: BGE-base-en-v1.5 via sentence-transformers
- **Vector DB**: FAISS (local and free)
- **Framework**: LangChain
- **Serving API**: FastAPI

## Prerequisites

- Python 3.8 or higher
- At least 8GB RAM (16GB recommended)
- 10GB free disk space
- Internet connection for initial setup

## Quick Start

### 1. Install Ollama

#### Windows
```bash
# Download from https://ollama.ai/download/windows
# Or use the setup script
python setup_ollama.py
```

#### macOS
```bash
brew install ollama
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Start Ollama Service
```bash
ollama serve
```

### 3. Pull Models
```bash
# Pull Llama3 (recommended)
ollama pull llama3

# Or pull other models
ollama pull mistral
ollama pull phi3
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
# Full evaluation
python main.py

# Interactive demo
python demo.py

# API server
python api_server.py
```

## Detailed Setup

### Step 1: Ollama Installation

#### Automatic Setup
```bash
python setup_ollama.py
```

This script will:
- Check if Ollama is installed
- Install Ollama if needed
- Start the Ollama service
- Pull required models
- Test the installation

#### Manual Setup

1. **Download Ollama**:
   - Windows: https://ollama.ai/download/windows
   - macOS: https://ollama.ai/download/mac
   - Linux: https://ollama.ai/download/linux

2. **Start Ollama**:
   ```bash
   ollama serve
   ```

3. **Pull Models**:
   ```bash
   ollama pull llama3
   ```

### Step 2: Python Environment

1. **Create Virtual Environment**:
   ```bash
   python -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Configuration

Edit `config.py` to customize:

```python
# LLM Configuration
OLLAMA_MODEL = "llama3"  # Options: "llama3", "mistral", "phi3"

# Embedding Configuration
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Vector Database
VECTOR_DB_PATH = "vector_db"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# Memory Configuration
MAX_MEMORY_TURNS = 4
```

## Usage Options

### 1. Command Line Interface

#### Full Application
```bash
python main.py
```

#### Interactive Demo
```bash
python demo.py
```

#### Component Testing
```bash
python test.py
```

#### API Server
```bash
python api_server.py
```

### 2. API Endpoints

When running the API server (`python api_server.py`):

- **Base URL**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Status**: http://localhost:8000/status
- **Query**: POST http://localhost:8000/query
- **Evaluation**: POST http://localhost:8000/evaluate

### 3. LangChain Integration

The application includes a LangChain-based implementation:

```python
from src.langchain_rag import LangChainRAG

# Initialize RAG system
rag = LangChainRAG()

# Process documents
rag.create_vectorstore_from_texts(texts)
rag.create_qa_chain()

# Query the system
result = rag.query("Your question here")
```

## Model Options

### LLM Models (Ollama)

1. **Llama3** (Recommended)
   - Size: ~4.7GB
   - Quality: Excellent
   - Speed: Good
   - Command: `ollama pull llama3`

2. **Mistral**
   - Size: ~4.1GB
   - Quality: Very Good
   - Speed: Excellent
   - Command: `ollama pull mistral`

3. **Phi-3-mini**
   - Size: ~2.3GB
   - Quality: Good
   - Speed: Excellent
   - Command: `ollama pull phi3`

### Embedding Models

1. **BGE-base-en-v1.5** (Default)
   - Size: ~440MB
   - Quality: Excellent
   - Speed: Good

2. **all-MiniLM-L6-v2**
   - Size: ~90MB
   - Quality: Good
   - Speed: Excellent

## Performance Optimization

### Hardware Requirements

- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores
- **GPU**: Optional but recommended for faster inference

### Optimization Tips

1. **Model Selection**:
   - Use Phi-3-mini for faster responses
   - Use Llama3 for better quality
   - Use Mistral for balanced performance

2. **Memory Management**:
   - Adjust chunk size based on available RAM
   - Use smaller embedding models for limited memory
   - Clear conversation memory regularly

3. **Batch Processing**:
   - Process documents in batches
   - Use background tasks for large operations
   - Cache embeddings when possible

## Troubleshooting

### Common Issues

1. **Ollama Not Running**:
   ```bash
   ollama serve
   ```

2. **Model Not Found**:
   ```bash
   ollama pull llama3
   ```

3. **Memory Issues**:
   - Reduce chunk size in config.py
   - Use smaller models
   - Close other applications

4. **Slow Performance**:
   - Use GPU if available
   - Reduce chunk size
   - Use smaller models

### Performance Monitoring

1. **Check Ollama Status**:
   ```bash
   ollama list
   ```

2. **Monitor Memory Usage**:
   ```bash
   # Linux/Mac
   top -p $(pgrep ollama)
   
   # Windows
   tasklist | findstr ollama
   ```

3. **API Health Check**:
   ```bash
   curl http://localhost:8000/status
   ```

## Advanced Configuration

### Custom Models

1. **Add Custom Model**:
   ```bash
   ollama create mymodel -f Modelfile
   ```

2. **Update Configuration**:
   ```python
   OLLAMA_MODEL = "mymodel"
   ```

### Custom Embeddings

1. **Use Different Embedding Model**:
   ```python
   EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
   ```

2. **Custom Embedding Pipeline**:
   ```python
   from sentence_transformers import SentenceTransformer
   
   model = SentenceTransformer("your-custom-model")
   ```

## Cost Comparison

### Free Stack vs Paid APIs

| Component | Free Stack | Paid APIs |
|-----------|------------|-----------|
| LLM | Ollama (Local) | OpenAI/Anthropic |
| Embeddings | BGE (Local) | OpenAI/Cohere |
| Vector DB | FAISS (Local) | Pinecone/Weaviate |
| **Total Cost** | **$0** | **$50-500/month** |

### Resource Usage

- **CPU**: Moderate usage during inference
- **RAM**: 4-8GB for models
- **Disk**: 5-10GB for models and data
- **Network**: Minimal after initial setup

## Next Steps

1. **Customize Models**: Try different Ollama models
2. **Optimize Performance**: Tune chunk sizes and parameters
3. **Add Features**: Implement custom evaluation metrics
4. **Scale Up**: Use multiple Ollama instances
5. **Deploy**: Set up production deployment

## Support

- **Ollama**: https://ollama.ai/docs
- **LangChain**: https://python.langchain.com/
- **FastAPI**: https://fastapi.tiangolo.com/
- **BGE**: https://huggingface.co/BAAI/bge-base-en-v1.5

## Conclusion

This free RAG stack provides excellent performance without any ongoing costs. The combination of Ollama, BGE embeddings, FAISS, LangChain, and FastAPI creates a robust, scalable solution for document-based question answering.

The setup is straightforward, and the system can be easily customized and extended based on your specific needs.
