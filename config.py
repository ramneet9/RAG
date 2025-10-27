# RAG Application Configuration - API-Based (Trial Friendly)

# PDF Sources
PDF_URLS = [
    "https://arxiv.org/pdf/1706.03762.pdf",  # Attention Is All You Need
    "https://arxiv.org/pdf/1810.04805.pdf",  # BERT
    "https://arxiv.org/pdf/2005.14165.pdf",  # GPT-3
    "https://arxiv.org/pdf/1907.11692.pdf",  # RoBERTa
    "https://arxiv.org/pdf/1910.10683.pdf"   # T5
]

# API Configuration - Choose your preferred provider
API_PROVIDER = "openai"  # Options: "openai", "anthropic", "huggingface", "cohere"

# OpenAI Configuration (Recommended for trials)
OPENAI_API_KEY = ""  # Set your OpenAI API key here
OPENAI_MODEL = "gpt-3.5-turbo"  # Cost-effective model
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # Cheapest embedding model

# Anthropic Configuration
ANTHROPIC_API_KEY = ""  # Set your Anthropic API key here
ANTHROPIC_MODEL = "claude-3-haiku-20240307"  # Cheapest Claude model

# Hugging Face Configuration (Free tier available)
HUGGINGFACE_API_KEY = ""  # Set your Hugging Face API key here
HUGGINGFACE_MODEL = "microsoft/DialoGPT-medium"
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cohere Configuration
COHERE_API_KEY = ""  # Set your Cohere API key here
COHERE_MODEL = "command-light"  # Cheapest Cohere model
COHERE_EMBEDDING_MODEL = "embed-english-v3.0"

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
