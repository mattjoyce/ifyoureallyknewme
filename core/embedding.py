"""
Embedding utility functions for TheInterview project.
Provides common functionality for generating and comparing text embeddings.
"""

import os
import numpy as np
import base64
from typing import List, Tuple, Union, Optional
from pathlib import Path
from openai import OpenAI
from .config import ConfigSchema

def get_embedding(config:ConfigSchema, text: str, model: Optional[str] = None) -> np.ndarray:
    """Get embedding from OpenAI API.
    
    Args:
        text: The text to embed
        model: Optional model name to use for embedding. If None, uses the model from config.
        
    Returns:
        Numpy array containing the embedding vector
    """

    
    if model is None:
        model = config.llm.embedding_model
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    response = client.embeddings.create(
        model=model,
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity (0-1)
    """
    # Ensure vectors are properly shaped
    v1 = v1.flatten()
    v2 = v2.flatten()
    
    # Handle zero vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(np.dot(v1, v2) / (norm1 * norm2))

def cosine_similarity_base64(a: str, b: str, dtype=np.float32) -> float:
    """Compute cosine similarity between two base64 encoded vectors.

    Args:
        a: Base64 encoded vector
        b: Base64 encoded vector
        dtype: The data type of the vector (default: np.float32)

    Returns:
        Cosine similarity (0-1)
    """
    try:
        a_bytes = base64.b64decode(a)
        b_bytes = base64.b64decode(b)
        a_array = np.frombuffer(a_bytes, dtype=dtype)
        b_array = np.frombuffer(b_bytes, dtype=dtype)
        return cosine_similarity(a_array, b_array)
    except (ValueError, TypeError) as e:
        print(f"Error computing cosine similarity: {e}")
        return 0.0 # Return 0 if there is an error.


def compute_pairwise_similarities(embeddings: List[np.ndarray]) -> np.ndarray:
    """Compute pairwise cosine similarities between multiple embeddings.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        2D numpy array where similarities[i][j] is the similarity between 
        embeddings[i] and embeddings[j]
    """
    if not embeddings:
        return np.array([])
        
    # Stack embeddings into a 2D array
    stacked = np.vstack(embeddings)
    
    # Normalize embeddings
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    normalized = stacked / norms
    
    # Compute similarities (dot products of normalized vectors)
    return np.dot(normalized, normalized.T)

def find_similar_texts(
    query_text: str, 
    corpus: List[str], 
    threshold: float = 0.85,
    model: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Find texts in a corpus that are similar to a query text.
    
    Args:
        query_text: The text to compare against
        corpus: List of texts to search through
        threshold: Similarity threshold (0-1)
        model: Optional model name to use for embedding

        
    Returns:
        List of (text, similarity) tuples for matches above threshold,
        sorted by descending similarity
    """
    if not corpus:
        return []
           
    model_name = model or EMBEDDING_MODEL
    
    # Get query embedding
    query_embedding = get_embedding(text=query_text, model=model_name)
    
    # Process corpus in batches to avoid API limits
    batch_size = 20
    all_similarities = []
    
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        batch_embeddings = [get_embedding(text, model_name, client) for text in batch]
        
        for j, emb in enumerate(batch_embeddings):
            similarity = cosine_similarity(query_embedding, emb)
            if similarity >= threshold:
                all_similarities.append((corpus[i+j], similarity))
    
    # Sort by similarity descending
    return sorted(all_similarities, key=lambda x: x[1], reverse=True)