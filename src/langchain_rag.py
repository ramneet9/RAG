"""
LangChain RAG Implementation

Advanced RAG implementation using LangChain framework.
"""

import os
from typing import List, Dict, Any
import logging
from pathlib import Path

from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, EMBEDDING_MODEL, 
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL, MAX_MEMORY_TURNS
)

logger = logging.getLogger(__name__)

class LangChainRAG:
    """Advanced RAG implementation using LangChain."""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LangChain components."""
        logger.info("Initializing LangChain RAG components...")
        
        # Initialize LLM (Ollama)
        self.llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.7
        )
        
        # Initialize embeddings (BGE)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=MAX_MEMORY_TURNS,
            memory_key="chat_history",
            return_messages=True
        )
        
        logger.info("LangChain components initialized successfully")
    
    def create_vectorstore_from_texts(self, texts: List[Dict[str, str]]) -> None:
        """
        Create vector store from texts using LangChain.
        
        Args:
            texts: List of text dictionaries
        """
        logger.info("Creating vector store with LangChain...")
        
        # Prepare documents
        documents = []
        for text_dict in texts:
            documents.append(text_dict["text"])
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Split documents
        splits = text_splitter.split_text("\n\n".join(documents))
        
        # Create vector store
        self.vectorstore = FAISS.from_texts(
            texts=splits,
            embedding=self.embeddings
        )
        
        logger.info(f"Created vector store with {len(splits)} chunks")
    
    def create_qa_chain(self) -> None:
        """Create QA chain with custom prompt."""
        
        # Custom prompt template
        prompt_template = """You are a helpful AI assistant that answers questions based on the provided context.

Context: {context}

Previous conversation:
{chat_history}

Question: {question}

Please provide a helpful answer based on the context. If the context doesn't contain enough information, say so clearly.

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": TOP_K_RETRIEVAL}
            ),
            memory=self.memory,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        logger.info("QA chain created successfully")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            
        Returns:
            Response with answer and metadata
        """
        try:
            result = self.qa_chain({"query": question})
            
            return {
                "question": question,
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "chat_history": self.memory.chat_memory.messages
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return {
                "question": question,
                "answer": "I apologize, but I encountered an error while processing your question.",
                "source_documents": [],
                "chat_history": []
            }
    
    def save_vectorstore(self, path: str) -> None:
        """Save vector store to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            logger.info(f"Vector store saved to {path}")
    
    def load_vectorstore(self, path: str) -> bool:
        """Load vector store from disk."""
        try:
            self.vectorstore = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            return False
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary."""
        return {
            "memory_type": "ConversationBufferWindowMemory",
            "window_size": MAX_MEMORY_TURNS,
            "current_messages": len(self.memory.chat_memory.messages),
            "messages": [
                {"role": msg.__class__.__name__, "content": msg.content}
                for msg in self.memory.chat_memory.messages
            ]
        }
