# Trial-Friendly RAG Setup Guide

## Overview

This guide helps you set up the RAG application using API-based services with trial accounts. This approach requires minimal disk space and setup time.

## Quick Start (5 minutes)

### 1. Get API Keys

#### OpenAI (Recommended for trials)
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create account and get $5 free credit
3. Generate API key

#### Alternative: Hugging Face (Free tier)
1. Visit [Hugging Face](https://huggingface.co/)
2. Create account (free)
3. Generate API token

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
```bash
python setup_api_keys.py
```

### 4. Run the Application
```bash
python main.py
```

## Detailed Setup

### Step 1: Choose Your Provider

#### Option A: OpenAI (Recommended)
- **Cost**: $5 free credit (enough for full evaluation)
- **Quality**: Excellent
- **Setup**: Easy

#### Option B: Hugging Face (Free)
- **Cost**: Free tier available
- **Quality**: Good
- **Setup**: Easy

#### Option C: Anthropic
- **Cost**: $5 free credit
- **Quality**: Excellent
- **Setup**: Easy

### Step 2: Get API Keys

#### OpenAI Setup
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up for an account
3. Go to API Keys section
4. Create new API key
5. Copy the key (starts with `sk-`)

#### Hugging Face Setup
1. Go to [Hugging Face](https://huggingface.co/)
2. Sign up for an account
3. Go to Settings > Access Tokens
4. Create new token
5. Copy the token

### Step 3: Configure the Application

#### Method 1: Interactive Setup
```bash
python setup_api_keys.py
```

#### Method 2: Manual Configuration
Edit `config.py`:
```python
# For OpenAI
API_PROVIDER = "openai"
OPENAI_API_KEY = "your_api_key_here"

# For Hugging Face
API_PROVIDER = "huggingface"
HUGGINGFACE_API_KEY = "your_token_here"
```

#### Method 3: Environment Variables
```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

#### Full Evaluation
```bash
python main.py
```

#### Interactive Demo
```bash
python demo.py
```

#### API Server
```bash
python api_server.py
```

## Cost Estimation

### OpenAI Trial
- **Free Credit**: $5
- **GPT-3.5-turbo**: ~$0.002 per 1K tokens
- **Embeddings**: ~$0.0001 per 1K tokens
- **Estimated Usage**: ~$2-3 for full evaluation

### Hugging Face
- **Free Tier**: 1000 requests/month
- **Cost**: $0 for free tier
- **Limitations**: Rate limits apply

### Anthropic Trial
- **Free Credit**: $5
- **Claude-3-haiku**: ~$0.0008 per 1K tokens
- **Estimated Usage**: ~$1-2 for full evaluation

## Trial-Friendly Features

### Cost Control
- **Token Limits**: Configurable max tokens per request
- **Batch Processing**: Smaller batches to avoid rate limits
- **Rate Limiting**: Built-in delays for trial accounts

### Resource Optimization
- **No Local Models**: No large downloads required
- **Minimal Disk Space**: Only ~500MB for dependencies
- **Fast Setup**: Ready in minutes

## Usage Examples

### Basic Usage
```python
from src.api_llm_client import APILLMClient
from src.vector_store import VectorStore

# Initialize clients
llm_client = APILLMClient()
vector_store = VectorStore()

# Process documents and query
# (See main.py for full example)
```

### API Usage
```bash
# Start API server
python api_server.py

# Query via API
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the Transformer architecture?"}'
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```bash
   python setup_api_keys.py
   ```

2. **Rate Limiting**
   - Wait a few minutes
   - Check your API usage limits
   - Use smaller batch sizes

3. **Insufficient Credits**
   - Check your account balance
   - Use cheaper models
   - Reduce token limits

4. **Connection Issues**
   - Check internet connection
   - Verify API endpoints
   - Check firewall settings

### Performance Tips

1. **Use Cheaper Models**
   - GPT-3.5-turbo instead of GPT-4
   - text-embedding-3-small instead of large
   - Claude-3-haiku instead of Claude-3-opus

2. **Optimize Requests**
   - Reduce chunk sizes
   - Limit conversation history
   - Use smaller batches

3. **Monitor Usage**
   - Check API usage dashboards
   - Set up billing alerts
   - Track token consumption

## Comparison: Trial vs Local

| Aspect | Trial (API) | Local (Ollama) |
|--------|-------------|----------------|
| Setup Time | 5 minutes | 30+ minutes |
| Disk Space | 500MB | 10GB+ |
| Cost | $0-5 | $0 |
| Performance | Excellent | Good |
| Internet Required | Yes | No |
| Rate Limits | Yes | No |

## Next Steps

1. **Complete Evaluation**: Run full 10-question evaluation
2. **Generate Report**: Create PDF evaluation report
3. **Customize**: Modify questions and parameters
4. **Scale Up**: Upgrade to paid plans if needed

## Support

- **OpenAI**: [OpenAI Help](https://help.openai.com/)
- **Hugging Face**: [HF Docs](https://huggingface.co/docs)
- **Anthropic**: [Anthropic Docs](https://docs.anthropic.com/)

## Conclusion

The trial-based approach provides an excellent way to test and evaluate the RAG application with minimal setup and cost. The $5 free credits from most providers are sufficient to run the complete evaluation and generate comprehensive reports.

This approach is perfect for:
- Quick testing and evaluation
- Limited disk space
- Temporary usage
- Cost-conscious development
