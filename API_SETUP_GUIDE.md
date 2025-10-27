# API-Based RAG Application Setup Guide

## Overview

The RAG application has been updated to use API-based services instead of downloading large models locally. This approach offers several advantages:

- **Faster Setup**: No need to download multi-GB model files
- **Better Performance**: Access to state-of-the-art models
- **Cost Efficiency**: Pay only for what you use
- **Easy Updates**: Always use the latest model versions

## Supported API Providers

### 1. OpenAI
- **Models**: GPT-3.5-turbo, GPT-4, GPT-4-turbo
- **Embeddings**: text-embedding-3-small, text-embedding-3-large
- **Cost**: Pay-per-token pricing
- **Best for**: High-quality responses, reliable service

### 2. Anthropic (Claude)
- **Models**: Claude-3-sonnet, Claude-3-opus
- **Embeddings**: Not available (use OpenAI for embeddings)
- **Cost**: Pay-per-token pricing
- **Best for**: Long context, reasoning tasks

### 3. Hugging Face
- **Models**: Various open-source models
- **Embeddings**: sentence-transformers models
- **Cost**: Free tier available, paid for production
- **Best for**: Open-source models, custom fine-tuning

### 4. Cohere
- **Models**: Command, Command-light
- **Embeddings**: embed-english-v3.0
- **Cost**: Pay-per-token pricing
- **Best for**: Business applications, multilingual support

## Setup Instructions

### Step 1: Choose Your API Provider

Edit `config.py` and set your preferred provider:

```python
API_PROVIDER = "openai"  # Options: "openai", "anthropic", "huggingface", "cohere"
```

### Step 2: Get API Keys

#### OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account and get API key
3. Add credits to your account

#### Anthropic
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create an account and get API key
3. Add credits to your account

#### Hugging Face
1. Visit [Hugging Face](https://huggingface.co/)
2. Create an account and get API key
3. Free tier available

#### Cohere
1. Visit [Cohere Platform](https://cohere.com/)
2. Create an account and get API key
3. Add credits to your account

### Step 3: Configure API Keys

#### Option A: Environment Variables (Recommended)
```bash
# Windows
set OPENAI_API_KEY=your_api_key_here
set ANTHROPIC_API_KEY=your_api_key_here
set HUGGINGFACE_API_KEY=your_api_key_here
set COHERE_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
export ANTHROPIC_API_KEY=your_api_key_here
export HUGGINGFACE_API_KEY=your_api_key_here
export COHERE_API_KEY=your_api_key_here
```

#### Option B: Direct Configuration
Edit `config.py` and add your API keys:

```python
# OpenAI Configuration
OPENAI_API_KEY = "your_openai_api_key_here"
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Anthropic Configuration
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
ANTHROPIC_MODEL = "claude-3-sonnet-20240229"

# Hugging Face Configuration
HUGGINGFACE_API_KEY = "your_huggingface_api_key_here"
HUGGINGFACE_MODEL = "microsoft/DialoGPT-medium"
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cohere Configuration
COHERE_API_KEY = "your_cohere_api_key_here"
COHERE_MODEL = "command"
COHERE_EMBEDDING_MODEL = "embed-english-v3.0"
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python main.py
```

## Configuration Examples

### OpenAI Configuration
```python
API_PROVIDER = "openai"
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

### Anthropic Configuration
```python
API_PROVIDER = "anthropic"
ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
# Note: Anthropic doesn't provide embeddings, so you'll need to use OpenAI for embeddings
```

### Mixed Configuration (Recommended)
```python
API_PROVIDER = "openai"  # For embeddings
OPENAI_API_KEY = "sk-..."
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Use Anthropic for responses
ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
```

## Cost Estimation

### OpenAI
- **GPT-3.5-turbo**: ~$0.002 per 1K tokens
- **GPT-4**: ~$0.03 per 1K tokens
- **Embeddings**: ~$0.0001 per 1K tokens

### Anthropic
- **Claude-3-sonnet**: ~$0.015 per 1K tokens
- **Claude-3-opus**: ~$0.075 per 1K tokens

### Cohere
- **Command**: ~$0.015 per 1K tokens
- **Embeddings**: ~$0.0001 per 1K tokens

### Hugging Face
- **Free tier**: Limited requests
- **Paid**: Varies by model

## Troubleshooting

### Common Issues

1. **API Key Not Found**:
   - Check environment variables
   - Verify API key in config.py
   - Ensure API key is valid

2. **Rate Limiting**:
   - Reduce batch size in vector_store.py
   - Add delays between requests
   - Upgrade to higher tier

3. **Model Not Available**:
   - Check model name spelling
   - Verify model availability in your region
   - Use alternative model

4. **Embedding Dimension Mismatch**:
   - Ensure consistent embedding model
   - Recreate vector database
   - Check model compatibility

### Performance Tips

1. **Batch Processing**: Process embeddings in batches
2. **Caching**: Cache embeddings to avoid re-computation
3. **Model Selection**: Choose appropriate model for your use case
4. **Error Handling**: Implement retry logic for API failures

## Security Best Practices

1. **Never commit API keys to version control**
2. **Use environment variables for API keys**
3. **Rotate API keys regularly**
4. **Monitor API usage and costs**
5. **Set up billing alerts**

## Migration from Local Models

If you were using the local model version:

1. **Backup your vector database**: Copy `vector_db/` folder
2. **Update configuration**: Set API provider and keys
3. **Recreate embeddings**: Run the application to regenerate embeddings
4. **Test functionality**: Verify all components work correctly

## Support

For issues with specific API providers:

- **OpenAI**: [OpenAI Help Center](https://help.openai.com/)
- **Anthropic**: [Anthropic Documentation](https://docs.anthropic.com/)
- **Hugging Face**: [Hugging Face Docs](https://huggingface.co/docs)
- **Cohere**: [Cohere Documentation](https://docs.cohere.com/)

## Next Steps

1. Choose your API provider
2. Get API keys
3. Configure the application
4. Run the evaluation
5. Monitor costs and performance
6. Optimize based on results
