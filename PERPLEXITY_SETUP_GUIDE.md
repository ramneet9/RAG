# Perplexity + Sentence-Transformers RAG Setup Guide

## Overview

This guide helps you set up the RAG application using:
- **LLM**: Perplexity API (sonar-small-chat)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS (local)

This combination provides excellent performance with minimal cost and setup complexity.

## Quick Start (3 minutes)

### 1. Get Perplexity API Key
1. Visit [Perplexity API](https://www.perplexity.ai/settings/api)
2. Sign up or log in
3. Generate API key
4. Copy the key

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
```bash
python setup_perplexity.py
```

### 4. Run the Application
```bash
python main.py
```

## Detailed Setup

### Step 1: Perplexity API Setup

#### Get API Key
1. **Visit**: [Perplexity API Settings](https://www.perplexity.ai/settings/api)
2. **Sign Up**: Create account if you don't have one
3. **Generate Key**: Click "Generate API Key"
4. **Copy Key**: Save the key (starts with `pplx-`)

#### Pricing
- **Free Tier**: Limited requests
- **Pro Tier**: $20/month for higher limits
- **Pay-per-use**: Available for occasional usage

### Step 2: Install Dependencies

#### Core Dependencies
```bash
pip install sentence-transformers>=2.2.2
pip install faiss-cpu>=1.7.4
pip install requests>=2.31.0
```

#### Full Installation
```bash
pip install -r requirements.txt
```

### Step 3: Configuration

#### Method 1: Interactive Setup
```bash
python setup_perplexity.py
```

#### Method 2: Manual Configuration
Edit `config.py`:
```python
# LLM / Chat provider
API_PROVIDER = "perplexity"
PERPLEXITY_API_KEY = "your_api_key_here"
PERPLEXITY_MODEL = "sonar-small-chat"
PERPLEXITY_API_BASE = "https://api.perplexity.ai"

# Embedding (retrieval) provider
EMBEDDER_PROVIDER = "sentence_transformers"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

#### Method 3: Environment Variables
```bash
# Windows
set PERPLEXITY_API_KEY=your_api_key_here

# Linux/Mac
export PERPLEXITY_API_KEY=your_api_key_here
```

### Step 4: Test Configuration

#### Test API Connection
```bash
python setup_perplexity.py test
```

#### Test Full System
```bash
python test.py
```

## Usage Examples

### Basic Usage
```python
from src.hybrid_llm_client import HybridLLMClient
from src.vector_store import VectorStore

# Initialize clients
llm_client = HybridLLMClient()
vector_store = VectorStore()

# Process documents and query
# (See main.py for full example)
```

### Interactive Demo
```bash
python demo.py
```

### Full Evaluation
```bash
python main.py
```

### API Server
```bash
python api_server.py
```

## Configuration Options

### Perplexity Models
- **sonar-small-chat**: Fast, cost-effective (default)
- **sonar-medium-chat**: Balanced performance
- **sonar-large-chat**: Best quality

### Embedding Models
- **all-MiniLM-L6-v2**: Fast, good quality (default)
- **all-mpnet-base-v2**: Better quality, slower
- **all-distilroberta-v1**: Fastest, basic quality

### Performance Tuning
```python
# config.py
CHUNK_SIZE = 512          # Adjust for your documents
CHUNK_OVERLAP = 50        # Overlap between chunks
TOP_K_RETRIEVAL = 5       # Number of chunks to retrieve
MAX_TOKENS_PER_REQUEST = 200  # Limit API costs
```

## Cost Analysis

### Perplexity API Costs
- **sonar-small-chat**: ~$0.20 per 1M tokens
- **sonar-medium-chat**: ~$0.60 per 1M tokens
- **sonar-large-chat**: ~$1.00 per 1M tokens

### Estimated Usage
- **Full Evaluation**: ~$0.50-1.00
- **Interactive Demo**: ~$0.10-0.20 per session
- **API Server**: ~$0.01-0.05 per query

### Embedding Costs
- **Sentence-Transformers**: Free (local)
- **Model Download**: ~90MB (one-time)

## Performance Comparison

| Component | Perplexity + ST | OpenAI + ST | Local Ollama |
|-----------|-----------------|-------------|--------------|
| **Setup Time** | 3 minutes | 5 minutes | 30+ minutes |
| **Disk Space** | 500MB | 500MB | 10GB+ |
| **Cost per Query** | $0.001 | $0.002 | $0 |
| **Response Quality** | Excellent | Excellent | Good |
| **Speed** | Fast | Fast | Medium |

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```bash
   python setup_perplexity.py
   ```

2. **Rate Limiting**
   - Check your Perplexity account limits
   - Reduce batch sizes
   - Add delays between requests

3. **Embedding Model Download**
   - First run downloads the model (~90MB)
   - Ensure stable internet connection
   - Check disk space

4. **Connection Issues**
   - Verify API key is correct
   - Check internet connection
   - Test with: `python setup_perplexity.py test`

### Performance Tips

1. **Optimize Chunk Size**
   - Smaller chunks: Better precision, more API calls
   - Larger chunks: Fewer API calls, less precision

2. **Batch Processing**
   - Process embeddings in batches
   - Use smaller batch sizes for rate limits

3. **Caching**
   - Vector database is cached locally
   - Reuse existing embeddings when possible

## Advanced Configuration

### Custom Models
```python
# Use different Perplexity model
PERPLEXITY_MODEL = "sonar-medium-chat"

# Use different embedding model
EMBEDDER_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

### Custom Prompts
```python
# Modify prompts in hybrid_llm_client.py
def _build_prompt(self, context, query, conversation_history):
    # Custom prompt logic here
    pass
```

### API Customization
```python
# Modify API parameters in hybrid_llm_client.py
payload = {
    "model": self.model,
    "messages": [...],
    "max_tokens": self.max_tokens,
    "temperature": 0.7,  # Adjust creativity
    "stream": False
}
```

## Monitoring and Analytics

### Usage Tracking
- Monitor API usage in Perplexity dashboard
- Track embedding generation locally
- Log query patterns and costs

### Performance Metrics
- Response time per query
- Embedding generation speed
- Vector search performance
- API cost per evaluation

## Security Best Practices

1. **API Key Security**
   - Never commit API keys to version control
   - Use environment variables
   - Rotate keys regularly

2. **Data Privacy**
   - Embeddings generated locally
   - Only queries sent to Perplexity API
   - No document content sent to external APIs

3. **Rate Limiting**
   - Implement proper rate limiting
   - Monitor usage patterns
   - Set up billing alerts

## Support and Resources

- **Perplexity API**: [Documentation](https://docs.perplexity.ai/)
- **Sentence-Transformers**: [Documentation](https://www.sbert.net/)
- **FAISS**: [Documentation](https://faiss.ai/)

## Conclusion

The Perplexity + Sentence-Transformers combination provides an excellent balance of:
- **Performance**: High-quality responses
- **Cost**: Reasonable API costs
- **Setup**: Quick and easy
- **Flexibility**: Easy to customize and extend

This setup is perfect for:
- Quick prototyping
- Cost-conscious development
- High-quality RAG applications
- Production deployments

The hybrid approach gives you the best of both worlds: free local embeddings with powerful cloud-based language generation.
