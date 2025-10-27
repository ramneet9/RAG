"""
Conversation Manager Module

Handles conversational memory and coordinates between vector store and LLM.
"""

from typing import List, Dict, Optional
import logging
from config import MAX_MEMORY_TURNS
from .vector_store import VectorStore
from .hybrid_llm_client import HybridLLMClient

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation state and coordinates RAG operations."""
    
    def __init__(self, llm_client: HybridLLMClient, vector_store: VectorStore):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.conversation_history: List[Dict[str, str]] = []
        
    def add_to_history(self, query: str, response: str) -> None:
        """
        Add a conversation turn to history.
        
        Args:
            query: User query
            response: Assistant response
        """
        self.conversation_history.append({
            "query": query,
            "response": response
        })
        
        # Keep only the last MAX_MEMORY_TURNS
        if len(self.conversation_history) > MAX_MEMORY_TURNS:
            self.conversation_history = self.conversation_history[-MAX_MEMORY_TURNS:]
        
        logger.info(f"Added to conversation history. Total turns: {len(self.conversation_history)}")
    
    def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve relevant context for the query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            Formatted context string
        """
        try:
            # Search for relevant chunks
            results = self.vector_store.search(query, k=top_k)
            
            if not results:
                return "No relevant context found."
            
            # Format context
            context_parts = []
            for i, result in enumerate(results, 1):
                context_parts.append(f"[Context {i} from {result['source']}]:")
                context_parts.append(result['text'])
                context_parts.append("")  # Empty line for readability
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return "Error retrieving context."
    
    def generate_response(self, query: str) -> Dict[str, str]:
        """
        Generate response for a query using RAG.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Get relevant context
            context = self.get_relevant_context(query)
            
            # Generate response using LLM
            response = self.llm_client.generate_response(
                context=context,
                query=query,
                conversation_history=self.conversation_history
            )
            
            # Add to conversation history
            self.add_to_history(query, response)
            
            return {
                "query": query,
                "response": response,
                "context": context,
                "conversation_turns": len(self.conversation_history)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_response = "I apologize, but I encountered an error while processing your request."
            self.add_to_history(query, error_response)
            
            return {
                "query": query,
                "response": error_response,
                "context": "",
                "conversation_turns": len(self.conversation_history),
                "error": str(e)
            }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get current conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history_summary(self) -> Dict[str, any]:
        """
        Get summary of conversation history.
        
        Returns:
            Dictionary with history statistics
        """
        return {
            "total_turns": len(self.conversation_history),
            "max_memory_turns": MAX_MEMORY_TURNS,
            "current_turns": len(self.conversation_history),
            "history": self.conversation_history
        }
