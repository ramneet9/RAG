# RAG Application

A comprehensive Retrieval-Augmented Generation (RAG) application that ingests content from PDF documents, creates a vector database for semantic retrieval, and powers a conversational bot with memory.

## Features

- **PDF Ingestion**: Downloads and extracts text from 5 research papers
- **Text Processing**: Advanced preprocessing and chunking for optimal embedding
- **Vector Database**: FAISS-based vector store for efficient similarity search
- **LLM Integration**: Open-source language model from Hugging Face
- **Conversational Memory**: Maintains context over the last 4 interactions
- **Evaluation Framework**: Custom metrics for system performance assessment
- **Comprehensive Reporting**: Detailed evaluation reports in JSON format

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Sources   │───▶│  PDF Processor  │───▶│  Text Chunker   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Evaluation    │◀───│ Conversation    │◀───│  Vector Store   │
│   Framework     │    │   Manager       │    │   (FAISS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   LLM Client    │
                       │ (Hugging Face)  │
                       └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download the project files
2. Run the setup script:

```bash
python setup.py
```

This will:
- Install all required dependencies
- Create necessary directories
- Verify Python version compatibility

### Manual Installation

If the setup script fails, install dependencies manually:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python main.py
```

This will:
1. Download the 5 PDF documents
2. Extract and process text content
3. Create vector embeddings and FAISS index
4. Run evaluation with 10 predefined questions
5. Generate a comprehensive evaluation report

### Individual Components

You can also use individual components:

```python
from src import PDFProcessor, VectorStore, LLMClient, ConversationManager

# Process PDFs
processor = PDFProcessor()
processor.download_pdfs(PDF_URLS)
texts = processor.extract_texts()

# Create vector store
vector_store = VectorStore()
vector_store.create_index(chunks)

# Initialize conversation manager
llm_client = LLMClient()
conversation_manager = ConversationManager(llm_client, vector_store)

# Generate response
result = conversation_manager.generate_response("Your question here")
```

## Configuration

Edit `config.py` to customize:

- PDF sources
- Model configurations
- Chunking parameters
- Evaluation questions
- Memory settings

## Evaluation Metrics

The system evaluates responses based on:

1. **Relevance** (30%): How well the answer addresses the question
2. **Accuracy** (30%): Correctness based on source PDFs
3. **Contextual Awareness** (20%): Effective use of conversation memory
4. **Response Quality** (20%): Language clarity and informativeness

## Project Structure

```
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── pdf_processor.py    # PDF downloading and text extraction
│   ├── text_chunker.py     # Text preprocessing and chunking
│   ├── vector_store.py     # FAISS vector database operations
│   ├── llm_client.py       # Hugging Face LLM integration
│   ├── conversation_manager.py # Conversation state management
│   └── evaluator.py        # Evaluation framework
├── data/                   # Downloaded PDF files
├── reports/               # Generated evaluation reports
├── vector_db/              # FAISS index and metadata
├── config.py               # Configuration settings
├── main.py                 # Main application entry point
├── setup.py                # Setup script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## PDF Sources

The application processes these research papers:

1. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) - Transformer architecture
2. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/pdf/1810.04805.pdf) - BERT model
3. [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf) - GPT-3
4. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf) - RoBERTa
5. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) - T5

## Evaluation Questions

The system is tested with 10 predefined questions covering:

- Transformer architecture contributions
- Model differences and innovations
- Performance comparisons
- Computational requirements
- Limitations and challenges

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: The application automatically detects and uses CPU if CUDA is unavailable
2. **Memory Issues**: Reduce chunk size in `config.py` if running out of memory
3. **Download Failures**: Check internet connection and PDF URL accessibility
4. **Model Loading**: Ensure sufficient disk space for model downloads

### Performance Optimization

- Use GPU if available for faster inference
- Adjust chunk size based on available memory
- Modify top-k retrieval for different accuracy/speed tradeoffs

## Contributing

This is an academic project. For improvements or bug fixes:

1. Identify the issue or enhancement
2. Modify the relevant module
3. Test thoroughly
4. Update documentation

## License

This project is for educational purposes. Please respect the licenses of the underlying models and libraries used.

## Acknowledgments

- Hugging Face for open-source models
- FAISS for vector similarity search
- PyMuPDF for PDF processing
- The research paper authors for their valuable work
