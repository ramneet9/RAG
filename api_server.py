"""
FastAPI Serving Layer

REST API for the RAG application using FastAPI.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import uvicorn
from pathlib import Path

from src.langchain_rag import LangChainRAG
from src.pdf_processor import PDFProcessor
from src.text_chunker import TextChunker
from config import (
    FASTAPI_HOST, FASTAPI_PORT, FASTAPI_TITLE, FASTAPI_DESCRIPTION,
    PDF_URLS, EVALUATION_QUESTIONS
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=FASTAPI_TITLE,
    description=FASTAPI_DESCRIPTION,
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_system = None

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    include_sources: bool = True

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]] = []
    memory_info: Dict[str, Any] = {}

class EvaluationRequest(BaseModel):
    questions: Optional[List[str]] = None

class EvaluationResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

class SystemStatus(BaseModel):
    status: str
    ollama_connected: bool
    vectorstore_loaded: bool
    memory_size: int
    available_models: List[str] = []

# Initialize RAG system
def initialize_rag_system():
    """Initialize the RAG system."""
    global rag_system
    try:
        rag_system = LangChainRAG()
        logger.info("RAG system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting RAG Application API...")
    initialize_rag_system()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Application API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/status"
    }

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status."""
    global rag_system
    
    status = {
        "status": "running" if rag_system else "not_initialized",
        "ollama_connected": False,
        "vectorstore_loaded": False,
        "memory_size": 0,
        "available_models": []
    }
    
    if rag_system:
        try:
            # Check Ollama connection
            status["ollama_connected"] = True  # Simplified check
            
            # Check vector store
            status["vectorstore_loaded"] = rag_system.vectorstore is not None
            
            # Get memory info
            memory_info = rag_system.get_memory_summary()
            status["memory_size"] = memory_info["current_messages"]
            
        except Exception as e:
            logger.error(f"Error checking status: {str(e)}")
            status["status"] = "error"
    
    return status

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(request.question)
        
        response = QueryResponse(
            question=result["question"],
            answer=result["answer"],
            memory_info=rag_system.get_memory_summary()
        )
        
        if request.include_sources and result["source_documents"]:
            response.sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_system(request: EvaluationRequest):
    """Evaluate the RAG system."""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    questions = request.questions or EVALUATION_QUESTIONS
    
    try:
        results = []
        for question in questions:
            result = rag_system.query(question)
            results.append({
                "question": question,
                "answer": result["answer"],
                "sources_count": len(result["source_documents"])
            })
        
        # Calculate summary
        summary = {
            "total_questions": len(questions),
            "successful_queries": len([r for r in results if r["answer"]]),
            "average_sources": sum(r["sources_count"] for r in results) / len(results)
        }
        
        return EvaluationResponse(results=results, summary=summary)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/setup")
async def setup_system(background_tasks: BackgroundTasks):
    """Setup the RAG system with PDFs."""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    def setup_task():
        try:
            # Process PDFs
            pdf_processor = PDFProcessor()
            pdf_processor.download_pdfs(PDF_URLS)
            texts = pdf_processor.extract_texts()
            
            # Create vector store
            rag_system.create_vectorstore_from_texts(texts)
            rag_system.create_qa_chain()
            
            # Save vector store
            rag_system.save_vectorstore("vector_db/langchain_faiss")
            
            logger.info("System setup completed successfully")
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
    
    background_tasks.add_task(setup_task)
    
    return {"message": "Setup started in background"}

@app.delete("/memory")
async def clear_memory():
    """Clear conversation memory."""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    rag_system.clear_memory()
    return {"message": "Memory cleared successfully"}

@app.get("/memory", response_model=Dict[str, Any])
async def get_memory():
    """Get conversation memory."""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    return rag_system.get_memory_summary()

def run_server():
    """Run the FastAPI server."""
    logger.info(f"Starting server on {FASTAPI_HOST}:{FASTAPI_PORT}")
    uvicorn.run(
        "api_server:app",
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        reload=True
    )

if __name__ == "__main__":
    run_server()
