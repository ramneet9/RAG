"""
RAG Application - Main Entry Point

This module provides the main interface for the RAG application.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from src.pdf_processor import PDFProcessor
from src.text_chunker import TextChunker
from src.vector_store import VectorStore
from src.api_llm_client import APILLMClient
from src.conversation_manager import ConversationManager
from src.evaluator import RAGEvaluator
from config import PDF_URLS, EVALUATION_QUESTIONS
import logging

def main():
    """Main function to run the RAG application."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/rag_app.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    print("🚀 Starting RAG Application...")
    logger.info("RAG Application started")
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        pdf_processor = PDFProcessor()
        text_chunker = TextChunker()
        vector_store = VectorStore()
        llm_client = APILLMClient()
        conversation_manager = ConversationManager(llm_client, vector_store)
        evaluator = RAGEvaluator(conversation_manager)
        
        # Process PDFs and create vector database
        print("📄 Processing PDFs...")
        logger.info("Starting PDF processing")
        pdf_processor.download_pdfs(PDF_URLS)
        texts = pdf_processor.extract_texts()
        
        if not texts:
            raise ValueError("No texts extracted from PDFs")
        
        print("✂️ Chunking texts...")
        logger.info("Starting text chunking")
        chunks = text_chunker.chunk_texts(texts)
        
        if not chunks:
            raise ValueError("No chunks created from texts")
        
        print("🔍 Creating vector database...")
        logger.info("Creating vector database")
        vector_store.create_index(chunks)
        
        # Run evaluation
        print("🧪 Running evaluation...")
        logger.info("Starting evaluation")
        results = evaluator.evaluate(EVALUATION_QUESTIONS)
        
        # Generate report
        print("📊 Generating report...")
        logger.info("Generating evaluation report")
        report_path = evaluator.generate_report(results)
        
        print("✅ RAG Application completed successfully!")
        print(f"📋 Report generated: {report_path}")
        logger.info(f"RAG Application completed successfully. Report: {report_path}")
        
    except Exception as e:
        error_msg = f"RAG Application failed: {str(e)}"
        print(f"❌ {error_msg}")
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    main()
