"""
Text Chunking Module

Handles text preprocessing and chunking for embedding generation.
"""

import re
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    """Handles text preprocessing and chunking."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}"\'-]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting based on punctuation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str, filename: str) -> List[Dict[str, str]]:
        """
        Create overlapping chunks from text.
        
        Args:
            text: Text to chunk
            filename: Source filename
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        sentences = self.split_into_sentences(text)
        
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    "chunk_id": f"{filename}_{chunk_id}",
                    "text": current_chunk.strip(),
                    "source": filename,
                    "chunk_index": chunk_id
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                "chunk_id": f"{filename}_{chunk_id}",
                "text": current_chunk.strip(),
                "source": filename,
                "chunk_index": chunk_id
            })
        
        return chunks
    
    def chunk_texts(self, texts: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Process and chunk all texts.
        
        Args:
            texts: List of text dictionaries
            
        Returns:
            List of chunk dictionaries
        """
        all_chunks = []
        
        for text_dict in texts:
            filename = text_dict["filename"]
            text = text_dict["text"]
            
            logger.info(f"Processing {filename}...")
            
            # Preprocess text
            cleaned_text = self.preprocess_text(text)
            
            # Create chunks
            chunks = self.create_chunks(cleaned_text, filename)
            
            all_chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from {filename}")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
