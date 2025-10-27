"""
API-based LLM Client Module

Handles integration with various LLM APIs instead of local models.
"""

import os
import requests
import json
from typing import List, Dict, Optional
import logging
from config import (
    API_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    HUGGINGFACE_API_KEY, HUGGINGFACE_MODEL, HUGGINGFACE_EMBEDDING_MODEL,
    COHERE_API_KEY, COHERE_MODEL, COHERE_EMBEDDING_MODEL,
    USE_TRIAL_MODE, MAX_TOKENS_PER_REQUEST, BATCH_SIZE
)

logger = logging.getLogger(__name__)

class APILLMClient:
    """Handles LLM integration using various API providers."""
    
    def __init__(self):
        self.provider = API_PROVIDER.lower()
        self.api_key = self._get_api_key()
        self.model = self._get_model()
        self.embedding_model = self._get_embedding_model()
        self.trial_mode = USE_TRIAL_MODE
        self.max_tokens = MAX_TOKENS_PER_REQUEST
        self.batch_size = BATCH_SIZE
        
        logger.info(f"Initialized API LLM client with provider: {self.provider}")
        logger.info(f"Trial mode: {self.trial_mode}, Max tokens: {self.max_tokens}")
        
        # Validate API key
        if not self.api_key:
            logger.warning(f"No API key found for {self.provider}. Please set your API key.")
        
    def _get_api_key(self) -> str:
        """Get API key based on provider."""
        if self.provider == "openai":
            return OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            return ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
        elif self.provider == "huggingface":
            return HUGGINGFACE_API_KEY or os.getenv("HUGGINGFACE_API_KEY")
        elif self.provider == "cohere":
            return COHERE_API_KEY or os.getenv("COHERE_API_KEY")
        else:
            raise ValueError(f"Unsupported API provider: {self.provider}")
    
    def _get_model(self) -> str:
        """Get model name based on provider."""
        if self.provider == "openai":
            return OPENAI_MODEL
        elif self.provider == "anthropic":
            return ANTHROPIC_MODEL
        elif self.provider == "huggingface":
            return HUGGINGFACE_MODEL
        elif self.provider == "cohere":
            return COHERE_MODEL
        else:
            raise ValueError(f"Unsupported API provider: {self.provider}")
    
    def _get_embedding_model(self) -> str:
        """Get embedding model name based on provider."""
        if self.provider == "openai":
            return OPENAI_EMBEDDING_MODEL
        elif self.provider == "huggingface":
            return HUGGINGFACE_EMBEDDING_MODEL
        elif self.provider == "cohere":
            return COHERE_EMBEDDING_MODEL
        else:
            raise ValueError(f"Unsupported API provider: {self.provider}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.provider == "openai":
            return self._openai_embeddings(texts)
        elif self.provider == "huggingface":
            return self._huggingface_embeddings(texts)
        elif self.provider == "cohere":
            return self._cohere_embeddings(texts)
        else:
            raise ValueError(f"Embeddings not supported for provider: {self.provider}")
    
    def _openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        import openai
        
        client = openai.OpenAI(api_key=self.api_key)
        
        try:
            response = client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI embeddings error: {str(e)}")
            raise
    
    def _huggingface_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Hugging Face API."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{self.embedding_model}",
                    headers=headers,
                    json={"inputs": text}
                )
                
                if response.status_code == 200:
                    embeddings.append(response.json())
                else:
                    logger.error(f"Hugging Face API error: {response.status_code}")
                    raise Exception(f"API error: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Hugging Face embeddings error: {str(e)}")
                raise
        
        return embeddings
    
    def _cohere_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Cohere API."""
        import cohere
        
        co = cohere.Client(self.api_key)
        
        try:
            response = co.embed(
                texts=texts,
                model=self.embedding_model
            )
            
            return response.embeddings
            
        except Exception as e:
            logger.error(f"Cohere embeddings error: {str(e)}")
            raise
    
    def generate_response(self, context: str, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate response using API.
        
        Args:
            context: Retrieved context from vector store
            query: User query
            conversation_history: Previous conversation turns
            
        Returns:
            Generated response
        """
        if self.provider == "openai":
            return self._openai_response(context, query, conversation_history)
        elif self.provider == "anthropic":
            return self._anthropic_response(context, query, conversation_history)
        elif self.provider == "huggingface":
            return self._huggingface_response(context, query, conversation_history)
        elif self.provider == "cohere":
            return self._cohere_response(context, query, conversation_history)
        else:
            raise ValueError(f"Unsupported API provider: {self.provider}")
    
    def _openai_response(self, context: str, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate response using OpenAI API."""
        import openai
        
        client = openai.OpenAI(api_key=self.api_key)
        
        # Build messages
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the provided context. Use the context information to provide accurate and relevant answers. If the context doesn't contain enough information, say so clearly."}
        ]
        
        # Add conversation history
        if conversation_history:
            for turn in conversation_history[-4:]:  # Last 4 turns
                messages.append({"role": "user", "content": turn["query"]})
                messages.append({"role": "assistant", "content": turn["response"]})
        
        # Add context and current query
        messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"})
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI response error: {str(e)}")
            raise
    
    def _anthropic_response(self, context: str, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate response using Anthropic API."""
        import anthropic
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        # Build prompt
        prompt = "You are a helpful AI assistant that answers questions based on the provided context.\n\n"
        
        # Add conversation history
        if conversation_history:
            prompt += "Previous conversation:\n"
            for turn in conversation_history[-4:]:
                prompt += f"Human: {turn['query']}\nAssistant: {turn['response']}\n"
            prompt += "\n"
        
        # Add context and query
        prompt += f"Context: {context}\n\nQuestion: {query}\n\nPlease provide a helpful answer based on the context."
        
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic response error: {str(e)}")
            raise
    
    def _huggingface_response(self, context: str, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate response using Hugging Face API."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Build prompt
        prompt = self._build_prompt(context, query, conversation_history)
        
        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json={"inputs": prompt}
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").replace(prompt, "").strip()
                else:
                    return str(result)
            else:
                logger.error(f"Hugging Face API error: {response.status_code}")
                raise Exception(f"API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Hugging Face response error: {str(e)}")
            raise
    
    def _cohere_response(self, context: str, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate response using Cohere API."""
        import cohere
        
        co = cohere.Client(self.api_key)
        
        # Build prompt
        prompt = self._build_prompt(context, query, conversation_history)
        
        try:
            response = co.generate(
                model=self.model,
                prompt=prompt,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.generations[0].text.strip()
            
        except Exception as e:
            logger.error(f"Cohere response error: {str(e)}")
            raise
    
    def _build_prompt(self, context: str, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Build prompt with context and conversation history."""
        prompt_parts = []
        
        # Add system instruction
        prompt_parts.append("You are a helpful AI assistant that answers questions based on the provided context.")
        prompt_parts.append("Use the context information to provide accurate and relevant answers.")
        prompt_parts.append("If the context doesn't contain enough information, say so clearly.")
        prompt_parts.append("")
        
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
        prompt_parts.append(f"Human: {query}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
