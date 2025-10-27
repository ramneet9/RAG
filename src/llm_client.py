"""
LLM Client Module

Handles integration with open-source language models from Hugging Face.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict
import logging
from config import LLM_MODEL

logger = logging.getLogger(__name__)

class LLMClient:
    """Handles LLM integration and response generation."""
    
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading LLM model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
    
    def generate_response(self, context: str, query: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate response using retrieved context and conversation history.
        
        Args:
            context: Retrieved context from vector store
            query: User query
            conversation_history: Previous conversation turns
            
        Returns:
            Generated response
        """
        # Build prompt with context and conversation history
        prompt = self._build_prompt(context, query, conversation_history)
        
        try:
            # Generate response
            response = self.generator(
                prompt,
                max_length=len(prompt.split()) + 150,  # Allow for response generation
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]["generated_text"]
            
            # Remove the original prompt to get only the response
            response_text = generated_text[len(prompt):].strip()
            
            # Clean up the response
            response_text = self._clean_response(response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
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
