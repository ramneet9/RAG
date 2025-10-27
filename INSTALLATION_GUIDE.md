# RAG Application - Installation and Usage Guide

## Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (8GB recommended)
- Internet connection for model downloads

### 2. Installation

```bash
# Clone or download the project files
# Navigate to the project directory

# Run comprehensive setup
python setup_environment.py

# Or run basic setup
python setup.py
```

### 3. Usage

#### Full Application (Recommended)
```bash
python main.py
```
This will:
- Download 5 PDF documents
- Process and chunk text content
- Create vector database
- Run evaluation with 10 questions
- Generate comprehensive PDF report

#### Interactive Demo
```bash
python demo.py
```
Choose between:
- Interactive mode (ask your own questions)
- Quick demo (predefined questions)

#### Component Testing
```bash
python test.py
```
Tests individual components to verify functionality.

#### Command Line Interface
```bash
# Full evaluation
python run.py full

# Interactive demo
python run.py interactive

# Quick demo
python run.py quick

# Component tests
python run.py test

# Environment setup
python run.py setup
```

## Detailed Setup Instructions

### Step 1: Environment Setup

1. **Check Python Version**:
   ```bash
   python --version
   ```
   Should be 3.8 or higher.

2. **Create Virtual Environment** (Recommended):
   ```bash
   python -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Directory Structure

The setup will create:
```
├── data/           # PDF files
├── reports/        # Evaluation reports
├── vector_db/      # FAISS index
├── logs/          # Application logs
└── src/           # Source code
```

### Step 3: Model Downloads

Models are downloaded automatically on first use:
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (~90MB)
- **LLM Model**: microsoft/DialoGPT-medium (~500MB)

## Configuration

Edit `config.py` to customize:

```python
# PDF Sources
PDF_URLS = [
    "https://arxiv.org/pdf/1706.03762.pdf",  # Transformer
    "https://arxiv.org/pdf/1810.04805.pdf",  # BERT
    "https://arxiv.org/pdf/2005.14165.pdf",  # GPT-3
    "https://arxiv.org/pdf/1907.11692.pdf",  # RoBERTa
    "https://arxiv.org/pdf/1910.10683.pdf"   # T5
]

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "microsoft/DialoGPT-medium"

# Vector Database Configuration
VECTOR_DB_PATH = "vector_db"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# Memory Configuration
MAX_MEMORY_TURNS = 4
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**:
   - The application automatically uses CPU if CUDA is unavailable
   - For GPU support, install PyTorch with CUDA support

2. **Memory Issues**:
   - Reduce `CHUNK_SIZE` in config.py
   - Close other applications
   - Use smaller models

3. **Download Failures**:
   - Check internet connection
   - Verify PDF URLs are accessible
   - Try downloading PDFs manually

4. **Model Loading Issues**:
   - Ensure sufficient disk space (~1GB)
   - Check internet connection for model downloads
   - Verify Hugging Face access

### Performance Optimization

1. **GPU Acceleration**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Memory Optimization**:
   - Reduce chunk size for lower memory usage
   - Use smaller embedding models
   - Process PDFs in batches

3. **Speed Optimization**:
   - Use GPU for faster inference
   - Increase chunk overlap for better retrieval
   - Cache vector database

## Evaluation Framework

The system evaluates responses on four dimensions:

1. **Relevance (30%)**: How well the answer addresses the question
2. **Accuracy (30%)**: Correctness based on source PDFs
3. **Contextual Awareness (20%)**: Effective use of conversation memory
4. **Response Quality (20%)**: Language clarity and informativeness

### Sample Evaluation Questions

1. What is the main contribution of the Transformer architecture?
2. How does BERT differ from previous language models?
3. What are the key innovations in GPT-3?
4. How does RoBERTa improve upon BERT?
5. What is the T5 model and how does it work?
6. What are attention mechanisms and why are they important?
7. How do these models handle different NLP tasks?
8. What are the computational requirements for these models?
9. How do these models compare in terms of performance?
10. What are the limitations of these transformer-based models?

## Output Files

### Reports
- **JSON Report**: `reports/rag_evaluation_report_YYYYMMDD_HHMMSS.json`
- **PDF Report**: `reports/rag_evaluation_report_YYYYMMDD_HHMMSS.pdf`

### Data Files
- **PDF Files**: `data/paper_1.pdf` to `data/paper_5.pdf`
- **Vector Database**: `vector_db/faiss_index.bin` and `vector_db/metadata.pkl`
- **Logs**: `logs/rag_app.log`

## Advanced Usage

### Custom Questions
Edit `config.py` to add your own evaluation questions:

```python
EVALUATION_QUESTIONS = [
    "Your custom question 1",
    "Your custom question 2",
    # ... more questions
]
```

### Custom Models
Change models in `config.py`:

```python
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Larger model
LLM_MODEL = "microsoft/DialoGPT-large"  # Larger model
```

### Custom PDFs
Add your own PDF URLs:

```python
PDF_URLS = [
    "https://your-domain.com/paper1.pdf",
    "https://your-domain.com/paper2.pdf",
    # ... more URLs
]
```

## Support

For issues or questions:
1. Check the logs in `logs/rag_app.log`
2. Run component tests with `python test.py`
3. Verify environment setup with `python setup_environment.py`
4. Check the README.md for additional information

## License

This project is for educational purposes. Please respect the licenses of the underlying models and libraries used.
