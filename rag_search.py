"""
RAG Search
Handles vector search using HNSW index and search term generation.
"""

import json
import numpy as np
import hnswlib
from typing import List, Dict, Any, Optional
from litellm_client import AsyncLiteLLMClient


class RAGSearch:
    """
    Manages HNSW vector index and provides RAG search functionality.
    """
    
    def __init__(self, client: AsyncLiteLLMClient):
        """
        Initialize RAG search.
        
        Args:
            client: AsyncLiteLLMClient instance
        """
        self.client = client
        self.index = None  # HNSW index
        self.chunks = []  # List of chunk dictionaries
        self.dimension = 1536  # Embedding dimension
        self.index_file = "index.json"
    
    def load_index(self, index_file: str = "index.json") -> None:
        """
        Load index from JSON file and build HNSW index.
        
        Args:
            index_file: Path to index JSON file
        """
        self.index_file = index_file
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Index file {index_file} not found. Run indexer.py first.")
        
        self.chunks = index_data["chunks"]
        
        if not self.chunks:
            raise ValueError("Index file contains no chunks")
        
        # Build HNSW index
        num_elements = len(self.chunks)
        
        # Initialize HNSW index for cosine similarity
        self.index = hnswlib.Index(space='cosine', dim=self.dimension)
        
        # Initialize with appropriate parameters
        # M: number of bi-directional links, ef_construction: size of dynamic candidate list
        self.index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        # Add embeddings to index
        embeddings = np.array([chunk["embedding"] for chunk in self.chunks], dtype=np.float32)
        labels = np.arange(num_elements)
        
        self.index.add_items(embeddings, labels)
        
        # Set ef parameter for search (should be >= k)
        self.index.set_ef(50)  # ef should be at least k (number of results to return)
    
    def rebuild_index(self) -> None:
        """
        Rebuild HNSW index from current chunks (after adding new chunks).
        """
        if not self.chunks:
            return
        
        num_elements = len(self.chunks)
        
        # Reinitialize index
        self.index = hnswlib.Index(space='cosine', dim=self.dimension)
        self.index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        # Add all embeddings
        embeddings = np.array([chunk["embedding"] for chunk in self.chunks], dtype=np.float32)
        labels = np.arange(num_elements)
        
        self.index.add_items(embeddings, labels)
        self.index.set_ef(50)
    
    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Add a new chunk to the index and rebuild HNSW.
        
        Args:
            chunk: Chunk dictionary with text, embedding, etc.
        """
        # Add to chunks list
        self.chunks.append(chunk)
        
        # Rebuild index
        self.rebuild_index()
        
        # Update index.json file
        with open(self.index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        index_data["chunks"] = self.chunks
        index_data["metadata"]["total_chunks"] = len(self.chunks)
        
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    async def generate_search_terms(
        self,
        user_question: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> List[str]:
        """
        Generate search terms from user question using GPT-5-nano.
        
        Args:
            user_question: User's question
            conversation_history: Recent conversation history
            
        Returns:
            List of 3-5 search terms
        """
        # Build context from conversation history
        history_text = ""
        if conversation_history:
            recent_history = conversation_history[-10:]  # Last 10 exchanges
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in recent_history
            ])
        
        prompt = f"""You are a search term generator for a Mass Effect lore database.

User question: {user_question}

{f'Recent conversation history:\n{history_text}' if history_text else ''}

Generate 3-5 short, specific search terms that would help find relevant information in a Mass Effect lore database to answer this question. Each search term should be a key phrase or concept (1-4 words).

Output only the search terms, one per line, without numbering or bullets.
"""
        
        response = await self.client.text_completion(
            prompt=prompt,
            model_name="azure.gpt-5-nano",
            max_tokens=20000,
            temperature=0.3
        )
        
        # Parse search terms (one per line)
        terms = [line.strip() for line in response.strip().split('\n') if line.strip()]
        
        # Limit to 5 terms
        return terms[:5]
    
    async def vector_search(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform vector search using HNSW index.
        
        Args:
            query_text: Text to search for
            k: Number of results to return
            
        Returns:
            List of chunk dictionaries (sorted by relevance)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Generate embedding for query
        embeddings = await self.client.embedding(
            input_texts=[query_text],
            model_name="azure.text-embedding-3-large",
            dimensions=1536
        )
        query_embedding = np.array(embeddings[0], dtype=np.float32)
        
        # Search in HNSW index
        labels, distances = self.index.knn_query(query_embedding, k=k)
        
        # Get chunks corresponding to labels
        results = []
        for label, distance in zip(labels[0], distances[0]):
            chunk = self.chunks[label].copy()
            chunk["similarity"] = 1 - distance  # Convert distance to similarity (cosine)
            results.append(chunk)
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results
    
    async def search_multiple_terms(
        self,
        search_terms: List[str],
        k_per_term: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for multiple terms and merge results.
        
        Args:
            search_terms: List of search terms
            k_per_term: Number of results per term
            
        Returns:
            Merged and deduplicated list of chunks
        """
        all_results = []
        seen_chunk_ids = set()
        
        for term in search_terms:
            results = await self.vector_search(term, k=k_per_term)
            for chunk in results:
                chunk_id = chunk["chunk_id"]
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_results.append(chunk)
        
        # Sort by similarity
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return all_results
    
    def get_chunk_texts(self, chunks: List[Dict[str, Any]], max_tokens: int = 8000) -> List[str]:
        """
        Extract text from chunks, respecting token limit.
        
        Args:
            chunks: List of chunk dictionaries
            max_tokens: Maximum total tokens
            
        Returns:
            List of chunk texts (limited by token count)
        """
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        
        texts = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = len(encoding.encode(chunk["text"]))
            if total_tokens + chunk_tokens > max_tokens:
                break
            texts.append(chunk["text"])
            total_tokens += chunk_tokens
        
        return texts

