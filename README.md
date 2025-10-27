# RAG Application - Perplexity + Sentence-Transformers

A comprehensive Retrieval-Augmented Generation (RAG) application that processes academic papers and provides intelligent conversational responses.

## Architecture

- **LLM**: Perplexity API (sonar-small-chat)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS (local)
- **Framework**: Custom implementation

## Quick Start

### 1. Get Perplexity API Key
1. Visit [Perplexity API Settings](https://www.perplexity.ai/settings/api)
2. Sign up/log in and generate API key
3. Copy the key (starts with `pplx-`)

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

## Usage

### Full Evaluation
```bash
python main.py
```

### Interactive Demo
```bash
python demo.py
```

### Component Testing
```bash
python test.py
```

## Project Structure

```
├── src/                    # Source code modules
│   ├── hybrid_llm_client.py    # Perplexity + sentence-transformers
│   ├── pdf_processor.py        # PDF processing
│   ├── text_chunker.py         # Text chunking
│   ├── vector_store.py         # FAISS operations
│   ├── conversation_manager.py # Memory management
│   ├── evaluator.py            # Evaluation framework
│   └── report_generator.py    # PDF report generation
├── data/                   # PDF files
├── reports/               # Evaluation reports
├── vector_db/             # FAISS index
├── logs/                  # Application logs
├── config.py              # Configuration
├── main.py                # Main application
├── demo.py                # Interactive demo
├── test.py                # Component tests
├── setup_perplexity.py    # Perplexity setup
├── requirements.txt       # Dependencies
├── README.md              # This file
└── PERPLEXITY_SETUP_GUIDE.md # Detailed setup guide
```

## Configuration

Edit `config.py` to customize:

```python
# LLM / Chat provider
API_PROVIDER = "perplexity"
PERPLEXITY_API_KEY = "your_api_key_here"
PERPLEXITY_MODEL = "sonar-small-chat"

# Embedding (retrieval) provider
EMBEDDER_PROVIDER = "sentence_transformers"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

## Features

- **PDF Processing**: Downloads and extracts text from 5 research papers
- **Vector Database**: FAISS-based similarity search
- **Conversational Memory**: 4-turn conversation history
- **Evaluation Framework**: 10 predefined questions with custom metrics
- **Report Generation**: Comprehensive PDF evaluation reports

## Cost Estimation

- **Perplexity API**: ~$0.20 per 1M tokens
- **Embeddings**: Free (local)
- **Full Evaluation**: ~$0.50-1.00
- **Per Query**: ~$0.001

## Support

- **Perplexity API**: [Documentation](https://docs.perplexity.ai/)
- **Sentence-Transformers**: [Documentation](https://www.sbert.net/)
- **Setup Guide**: See `PERPLEXITY_SETUP_GUIDE.md`