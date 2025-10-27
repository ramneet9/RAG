"""
Vector Store Module

Handles vector database creation and similarity search using FAISS with API-based embeddings.
"""

import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import logging
from pathlib import Path
from config import VECTOR_DB_PATH, TOP_K_RETRIEVAL, BATCH_SIZE, USE_TRIAL_MODE
from .hybrid_llm_client import HybridLLMClient

logger = logging.getLogger(__name__)

class VectorStore:
    """Handles vector database operations using FAISS."""
    
    def __init__(self, db_path: str = VECTOR_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize hybrid embedding client
        logger.info("Initializing hybrid embedding client")
        self.embedding_client = HybridLLMClient()
        
        # Initialize FAISS index
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def generate_embeddings(self, chunks: List[Dict[str, str]]) -> np.ndarray:
        """
        Generate embeddings for text chunks using API.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk["text"] for chunk in chunks]
        
        logger.info(f"Generating API embeddings for {len(texts)} chunks...")
        
        try:
            # Generate embeddings in batches to avoid API limits
            all_embeddings = []
            
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1)//BATCH_SIZE}")
                
                batch_embeddings = self.embedding_client.generate_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Add delay in trial mode to respect rate limits
                if USE_TRIAL_MODE:
                    import time
                    time.sleep(1)
            
            embeddings = np.array(all_embeddings)
            logger.info(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def create_index(self, chunks: List[Dict[str, str]]) -> None:
        """
        Create FAISS index from chunks.
        
        Args:
            chunks: List of chunk dictionaries
        """
        logger.info("Creating vector database index...")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks and metadata
        self.chunks = chunks
        self.metadata = [{"chunk_id": chunk["chunk_id"], "source": chunk["source"]} 
                         for chunk in chunks]
        
        # Save index and metadata
        self.save_index()
        
        logger.info(f"Index created with {self.index.ntotal} vectors")
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        index_path = self.db_path / "faiss_index.bin"
        metadata_path = self.db_path / "metadata.pkl"
        
        faiss.write_index(self.index, str(index_path))
        
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                "chunks": self.chunks,
                "metadata": self.metadata
            }, f)
        
        logger.info(f"Index saved to {index_path}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = self.db_path / "faiss_index.bin"
        metadata_path = self.db_path / "metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            return False
        
        try:
            self.index = faiss.read_index(str(index_path))
            
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.metadata = data["metadata"]
            
            logger.info(f"Index loaded with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return False
    
    def search(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, str]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        if self.index is None:
            if not self.load_index():
                raise ValueError("No index available. Please create index first.")
        
        # Generate query embedding
        query_embeddings = self.embedding_client.generate_embeddings([query])
        query_embedding = np.array(query_embeddings[0]).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                chunk = self.chunks[idx]
                results.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score)
                })
        
        return results
