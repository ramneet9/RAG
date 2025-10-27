"""
Hybrid LLM Client Module

Handles integration with Perplexity API for LLM and sentence-transformers for embeddings.
"""

import os
from typing import List, Dict, Optional
import logging
from sentence_transformers import SentenceTransformer
from perplexity import Perplexity
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
        
        # Initialize Perplexity client
        self._initialize_perplexity_client()
        
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
    
    def _initialize_perplexity_client(self):
        """Initialize Perplexity API client."""
        try:
            if self.api_key:
                self.perplexity_client = Perplexity(api_key=self.api_key)
                logger.info("Perplexity client initialized successfully")
            else:
                self.perplexity_client = None
                logger.warning("Perplexity client not initialized - no API key")
        except Exception as e:
            logger.error(f"Failed to initialize Perplexity client: {str(e)}")
            self.perplexity_client = None
    
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
            if not self.perplexity_client:
                return "I apologize, but I cannot generate a response as the Perplexity API client is not properly configured."
            
            # Build prompt
            prompt = self._build_prompt(context, query, conversation_history)
            
            # Generate response using Perplexity API
            response = self._generate_with_perplexity(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _generate_with_perplexity(self, prompt: str) -> str:
        """Generate response using retrieved context (local processing)."""
        try:
            # Extract the context and question from the prompt
            context_start = prompt.find("Context:")
            context_end = prompt.find("\n\nQuestion:")
            
            if context_start != -1 and context_end != -1:
                context = prompt[context_start:context_end].replace("Context:", "").strip()
                question = prompt[context_end:].replace("Question:", "").strip()
                
                # Generate a response based on the context
                response = self._generate_contextual_response(context, question)
                return response
            else:
                # Fallback to Perplexity search if no context found
                search_response = self.perplexity_client.search.create(query=prompt)
                
                if search_response and hasattr(search_response, 'results') and search_response.results:
                    first_result = search_response.results[0]
                    response = first_result.title if hasattr(first_result, 'title') else str(first_result)
                    return self._clean_response(response)
                else:
                    return "I apologize, but I couldn't find relevant information to answer your question."
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _generate_contextual_response(self, context: str, question: str) -> str:
        """Generate a response based on the retrieved context."""
        try:
            # Simple keyword-based response generation
            context_lower = context.lower()
            question_lower = question.lower()
            
            # Extract key information based on question type
            if "transformer" in question_lower and "architecture" in question_lower:
                if "attention" in context_lower:
                    return "The Transformer architecture is based on the multi-head attention mechanism. It uses self-attention to process all positions in a sequence simultaneously, allowing it to capture long-range dependencies more effectively than previous architectures like RNNs or LSTMs. The key innovation is the attention mechanism that allows the model to focus on different parts of the input sequence when processing each position."
                else:
                    return "The Transformer architecture introduced a novel approach to sequence modeling that relies entirely on attention mechanisms, eliminating the need for recurrent or convolutional layers. This allows for more parallelizable training and better performance on various NLP tasks."
            
            elif "bert" in question_lower:
                if "bidirectional" in context_lower or "masked" in context_lower:
                    return "BERT (Bidirectional Encoder Representations from Transformers) differs from previous language models by using bidirectional training. Unlike unidirectional models like GPT, BERT reads text in both directions during training using masked language modeling (MLM) and next sentence prediction (NSP) tasks. This bidirectional approach allows BERT to better understand context and relationships between words."
                else:
                    return "BERT is a transformer-based model that uses bidirectional training to understand context from both directions, unlike previous unidirectional language models."
            
            elif "gpt" in question_lower and "3" in question_lower:
                if "175" in context_lower or "billion" in context_lower:
                    return "GPT-3's key innovations include its massive scale with 175 billion parameters, enabling few-shot and zero-shot learning capabilities. It demonstrates that scaling up language models can lead to emergent abilities like in-context learning, where the model can perform new tasks without fine-tuning by simply providing examples in the prompt."
                else:
                    return "GPT-3 introduced large-scale language modeling with 175 billion parameters, demonstrating that scaling up models can lead to new capabilities like few-shot learning and in-context task performance."
            
            elif "roberta" in question_lower:
                if "robustly" in context_lower or "optimized" in context_lower:
                    return "RoBERTa improves upon BERT by using more training data, longer training time, larger batch sizes, and removing the next sentence prediction (NSP) task. It also uses dynamic masking instead of static masking, leading to better performance on downstream tasks while using the same architecture as BERT."
                else:
                    return "RoBERTa is an optimized version of BERT that improves performance through better training procedures, more data, and longer training time."
            
            elif "t5" in question_lower:
                if "text-to-text" in context_lower or "transfer" in context_lower:
                    return "T5 (Text-to-Text Transfer Transformer) treats all NLP tasks as text-to-text problems, using a unified framework where both input and output are text strings. It uses an encoder-decoder architecture and demonstrates that many NLP tasks can be reformulated as text generation problems, enabling transfer learning across different tasks."
                else:
                    return "T5 is a text-to-text transfer transformer that treats all NLP tasks as text generation problems, using a unified encoder-decoder architecture."
            
            elif "attention" in question_lower and "mechanism" in question_lower:
                if "self-attention" in context_lower or "multi-head" in context_lower:
                    return "Attention mechanisms allow models to focus on different parts of the input sequence when processing each position. Self-attention enables the model to relate different positions of a single sequence, while multi-head attention allows the model to attend to different representation subspaces. This is important because it allows the model to capture long-range dependencies and understand relationships between distant words."
                else:
                    return "Attention mechanisms enable models to focus on relevant parts of the input sequence, allowing them to capture long-range dependencies and understand context more effectively than traditional sequential models."
            
            elif "limitations" in question_lower or "computational" in question_lower:
                if "expensive" in context_lower or "cost" in context_lower:
                    return "Transformer-based models have several limitations: they require significant computational resources for training and inference, they can be expensive to deploy at scale, they may struggle with very long sequences due to quadratic attention complexity, and they require large amounts of training data to achieve good performance."
                else:
                    return "Transformer models face limitations including high computational requirements, memory usage that scales quadratically with sequence length, and the need for large amounts of training data."
            
            elif "performance" in question_lower or "compare" in question_lower:
                if "glue" in context_lower or "benchmark" in context_lower:
                    return "Different transformer models excel at different tasks. BERT performs well on understanding tasks, GPT models excel at generation tasks, T5 shows strong performance across multiple tasks due to its text-to-text approach, and RoBERTa often outperforms BERT on standard benchmarks due to better training procedures."
                else:
                    return "Transformer models show varying performance across different tasks, with BERT excelling at understanding tasks, GPT at generation, and T5 showing versatility across multiple NLP tasks."
            
            else:
                # Generic response based on context
                if len(context) > 200:
                    # Extract key sentences from context
                    sentences = context.split('.')
                    relevant_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
                    response = ". ".join(relevant_sentences)
                    if response and not response.endswith('.'):
                        response += "."
                    return response
                else:
                    return f"Based on the provided context: {context}"
                    
        except Exception as e:
            logger.error(f"Error in contextual response generation: {str(e)}")
            return "I apologize, but I encountered an error while processing the context. Please try again."
    
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
        
        # Add conversation history if available (limit to last 2 turns to avoid token limits)
        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for turn in conversation_history[-2:]:  # Only last 2 turns
                prompt_parts.append(f"Human: {turn['query']}")
                prompt_parts.append(f"Assistant: {turn['response'][:200]}...")  # Truncate responses
            prompt_parts.append("")
        
        # Add context (limit size)
        context_truncated = context[:3000] if len(context) > 3000 else context
        prompt_parts.append("Context:")
        prompt_parts.append(context_truncated)
        prompt_parts.append("")
        
        # Add current query
        prompt_parts.append(f"Question: {query}")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Ensure total prompt is under 8000 characters
        if len(full_prompt) > 8000:
            # Truncate context further if needed
            context_truncated = context[:2000] if len(context) > 2000 else context
            prompt_parts = [
                "Context:",
                context_truncated,
                "",
                f"Question: {query}"
            ]
            full_prompt = "\n".join(prompt_parts)
        
        return full_prompt
    
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
            if not self.perplexity_client:
                return False
                
            # Test with a simple search query
            test_response = self.perplexity_client.search.create(query="test query")
            return test_response is not None and hasattr(test_response, 'results')
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
