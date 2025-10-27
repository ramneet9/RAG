# RAG Application Configuration - Perplexity + Sentence-Transformers

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PDF Sources
PDF_URLS = [
    "https://arxiv.org/pdf/1706.03762.pdf",  # Attention Is All You Need
    "https://arxiv.org/pdf/1810.04805.pdf",  # BERT
    "https://arxiv.org/pdf/2005.14165.pdf",  # GPT-3
    "https://arxiv.org/pdf/1907.11692.pdf",  # RoBERTa
    "https://arxiv.org/pdf/1910.10683.pdf"   # T5
]

# Embedding (retrieval) provider
EMBEDDER_PROVIDER = "sentence_transformers"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM / Chat provider
API_PROVIDER = "perplexity"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")  # Load from environment variable
PERPLEXITY_MODEL = "llama-3.1-sonar-small-128k-online"
PERPLEXITY_API_BASE = "https://api.perplexity.ai"

# Validate API key is loaded
if not PERPLEXITY_API_KEY:
    raise ValueError("PERPLEXITY_API_KEY not found in environment variables. Please create a .env file with your API key.")

# Vector Database Configuration
VECTOR_DB_PATH = "vector_db"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# Memory Configuration
MAX_MEMORY_TURNS = 4

# Trial-Friendly Settings
USE_TRIAL_MODE = True
MAX_TOKENS_PER_REQUEST = 200  # Limit tokens for cost control
BATCH_SIZE = 10  # Smaller batches for API limits

# Evaluation Configuration
EVALUATION_QUESTIONS = [
    "What is the main contribution of the Transformer architecture?",
    "How does BERT differ from previous language models?",
    "What are the key innovations in GPT-3?",
    "How does RoBERTa improve upon BERT?",
    "What is the T5 model and how does it work?",
    "What are attention mechanisms and why are they important?",
    "How do these models handle different NLP tasks?",
    "What are the computational requirements for these models?",
    "How do these models compare in terms of performance?",
    "What are the limitations of these transformer-based models?"
]
