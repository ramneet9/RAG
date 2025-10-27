"""
RAG Application Package

A comprehensive RAG (Retrieval-Augmented Generation) application that:
- Ingests content from PDF documents
- Creates a vector database for semantic retrieval
- Powers a conversational bot with memory
- Evaluates system performance
"""

from .pdf_processor import PDFProcessor
from .text_chunker import TextChunker
from .vector_store import VectorStore
from .hybrid_llm_client import HybridLLMClient
from .conversation_manager import ConversationManager
from .evaluator import RAGEvaluator

__version__ = "1.0.0"
__author__ = "RAG Application Team"

__all__ = [
    "PDFProcessor",
    "TextChunker", 
    "VectorStore",
    "HybridLLMClient",
    "ConversationManager",
    "RAGEvaluator"
]
