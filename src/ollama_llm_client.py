"""
Ollama LLM Client Module

Handles integration with Ollama for local LLM inference.
"""

import requests
import json
from typing import List, Dict, Optional
import logging
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

class OllamaLLMClient:
    """Handles LLM integration using Ollama."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model
        
        logger.info(f"Initialized Ollama LLM client with model: {model}")
        logger.info(f"Ollama base URL: {base_url}")
        
        # Check if Ollama is running
        self._check_ollama_connection()
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                if self.model in model_names:
                    logger.info(f"✅ Model {self.model} is available")
                    return True
                else:
                    logger.warning(f"⚠️ Model {self.model} not found. Available models: {model_names}")
                    logger.info("Run: ollama pull llama3")
                    return False
            else:
                logger.error(f"❌ Ollama connection failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Cannot connect to Ollama: {str(e)}")
            logger.info("Make sure Ollama is running: ollama serve")
            return False
    
    def generate_response(self, context: str, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate response using Ollama.
        
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
            
            # Generate response
            response = self._generate_with_ollama(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate response using Ollama API."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 500
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "").strip()
            
            # Clean up the response
            generated_text = self._clean_response(generated_text)
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {str(e)}")
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
            if line.startswith(('Human:', 'Assistant:', 'Context:', 'Previous conversation:')):
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
    
    def list_available_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                logger.error(f"Failed to list models: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            url = f"{self.base_url}/api/pull"
            payload = {"name": model_name}
            
            response = requests.post(url, json=payload, timeout=300)  # 5 minutes timeout
            response.raise_for_status()
            
            logger.info(f"Successfully pulled model: {model_name}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to pull model {model_name}: {str(e)}")
            return False
