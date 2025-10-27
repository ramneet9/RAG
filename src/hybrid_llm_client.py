"""
Hybrid LLM Client Module

Handles integration with Perplexity API for LLM and sentence-transformers for embeddings.
"""

import os
import requests
import json
from typing import List, Dict, Optional
import logging
from sentence_transformers import SentenceTransformer
from config import (
    API_PROVIDER, PERPLEXITY_API_KEY, PERPLEXITY_MODEL, PERPLEXITY_API_BASE,
    EMBEDDER_PROVIDER, EMBEDDER_MODEL,
    USE_TRIAL_MODE, MAX_TOKENS_PER_REQUEST, BATCH_SIZE
)

logger = logging.getLogger(__name__)

class HybridLLMClient:
    """Handles LLM integration using Perplexity API and sentence-transformers."""
    
    def __init__(self):
        self.provider = API_PROVIDER.lower()
        self.api_key = self._get_api_key()
        self.model = PERPLEXITY_MODEL
        self.api_base = PERPLEXITY_API_BASE
        
        # Embedding configuration
        self.embedder_provider = EMBEDDER_PROVIDER.lower()
        self.embedder_model = EMBEDDER_MODEL
        
        # Trial settings
        self.trial_mode = USE_TRIAL_MODE
        self.max_tokens = MAX_TOKENS_PER_REQUEST
        self.batch_size = BATCH_SIZE
        
        logger.info(f"Initialized Hybrid LLM client with provider: {self.provider}")
        logger.info(f"Embedding provider: {self.embedder_provider}")
        logger.info(f"Trial mode: {self.trial_mode}, Max tokens: {self.max_tokens}")
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Validate API key
        if not self.api_key:
            logger.warning(f"No API key found for {self.provider}. Please set your API key.")
    
    def _get_api_key(self) -> str:
        """Get API key."""
        return PERPLEXITY_API_KEY or os.getenv("PERPLEXITY_API_KEY")
    
    def _initialize_embedding_model(self):
        """Initialize sentence-transformers embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.embedder_model}")
            self.embedding_model = SentenceTransformer(self.embedder_model)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using sentence-transformers.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using sentence-transformers...")
            
            # Generate embeddings in batches
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
                
                batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=True)
                all_embeddings.extend(batch_embeddings.tolist())
            
            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_response(self, context: str, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate response using Perplexity API.
        
        Args:
            context: Retrieved context from vector store
            query: User query
            conversation_history: Previous conversation turns
            
        Returns:
            Generated response
        """
        try:
            # Build prompt
            prompt = self._build_prompt(context, query, conversation_history)
            
            # Generate response using Perplexity API
            response = self._generate_with_perplexity(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _generate_with_perplexity(self, prompt: str) -> str:
        """Generate response using Perplexity API."""
        url = f"{self.api_base}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that answers questions based on the provided context. Use the context information to provide accurate and relevant answers. If the context doesn't contain enough information, say so clearly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "stream": False
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"].strip()
            
            # Clean up the response
            generated_text = self._clean_response(generated_text)
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Perplexity API error: {str(e)}")
            raise
    
    def _build_prompt(self, context: str, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Build prompt with context and conversation history.
        
        Args:
            context: Retrieved context
            query: Current query
            conversation_history: Previous conversation turns
            
        Returns:
            Formatted prompt
        """
        prompt_parts = []
        
        # Add conversation history if available
        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for turn in conversation_history[-4:]:  # Last 4 turns
                prompt_parts.append(f"Human: {turn['query']}")
                prompt_parts.append(f"Assistant: {turn['response']}")
            prompt_parts.append("")
        
        # Add context
        prompt_parts.append("Context:")
        prompt_parts.append(context)
        prompt_parts.append("")
        
        # Add current query
        prompt_parts.append(f"Question: {query}")
        
        return "\n".join(prompt_parts)
    
    def _clean_response(self, response: str) -> str:
        """
        Clean and format the generated response.
        
        Args:
            response: Raw generated response
            
        Returns:
            Cleaned response
        """
        # Remove any remaining prompt artifacts
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that look like prompts or system messages
            if line.startswith(('Human:', 'Assistant:', 'Context:', 'Previous conversation:', 'Question:')):
                break
            if line and not line.startswith('You are'):
                cleaned_lines.append(line)
        
        # Join lines and clean up
        cleaned_response = ' '.join(cleaned_lines)
        
        # Remove excessive whitespace
        cleaned_response = ' '.join(cleaned_response.split())
        
        # Ensure response ends properly
        if not cleaned_response.endswith(('.', '!', '?')):
            cleaned_response += '.'
        
        return cleaned_response
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            url = f"{self.api_base}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
