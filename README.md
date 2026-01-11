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

**Option A: Using Environment Variables (Recommended)**
```bash
# Copy the example file
copy env_example.txt .env

# Edit .env and add your API key
# PERPLEXITY_API_KEY=your_actual_api_key_here
```

**Option B: Using Setup Script**
```bash
python setup_perplexity.py
```

### 4. Run the Application

**Option A: Streamlit Web UI (Recommended)**
```bash
streamlit run streamlit_app.py
```
Then open your browser to `http://localhost:8501`

**Note:** Always use `localhost:8501` in your browser, even if the terminal shows `0.0.0.0`. The `0.0.0.0` is a server binding address, not a browser URL.

**Option B: Command Line Interface**
```bash
python main.py
```

## Usage

### Streamlit Web UI (Recommended)
```bash
streamlit run streamlit_app.py
```
Features:
- Interactive chat interface
- PDF upload and management
- System status monitoring
- Real-time validation
- AWS EC2 deployment ready

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
├── streamlit_app.py       # Streamlit web UI
├── setup_perplexity.py    # Perplexity setup
├── requirements.txt       # Dependencies
├── README.md              # This file
├── AWS_EC2_DEPLOYMENT.md  # AWS deployment guide
└── PERPLEXITY_SETUP_GUIDE.md # Detailed setup guide
```

## Configuration

### Environment Variables (Recommended)
Create a `.env` file with your API key:
```bash
PERPLEXITY_API_KEY=your_actual_api_key_here
```

### Direct Configuration
Edit `config.py` to customize other settings:

```python
# LLM / Chat provider
API_PROVIDER = "perplexity"
PERPLEXITY_MODEL = "llama-3.1-sonar-small-128k-online"

# Embedding (retrieval) provider
EMBEDDER_PROVIDER = "sentence_transformers"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

**Note**: API keys are now loaded from environment variables for security.

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

## Deployment

### AWS EC2 Deployment
See `AWS_EC2_DEPLOYMENT.md` for detailed instructions on deploying to AWS EC2.

Quick start:
1. Follow the deployment guide
2. Configure security groups (port 8501)
3. Run `streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0`

## Support

- **Perplexity API**: [Documentation](https://docs.perplexity.ai/)
- **Sentence-Transformers**: [Documentation](https://www.sbert.net/)
- **Streamlit**: [Documentation](https://docs.streamlit.io/)
- **Setup Guide**: See `PERPLEXITY_SETUP_GUIDE.md`
- **Deployment Guide**: See `AWS_EC2_DEPLOYMENT.md`