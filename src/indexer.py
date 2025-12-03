"""
Index Builder
Builds vector index from database.txt using rolling window chunking.
"""

import json
import tiktoken
from typing import List, Dict, Any
from .litellm_client import AsyncLiteLLMClient


class Indexer:
    """
    Builds and manages the vector index from source text files.
    """
    
    def __init__(self, client: AsyncLiteLLMClient):
        """
        Initialize the indexer.
        
        Args:
            client: AsyncLiteLLMClient instance for embeddings
        """
        self.client = client
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = 256
        self.overlap = 128
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using tiktoken.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        return self.encoding.encode(text)
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text using rolling window approach.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries with text, token info, and position
        """
        tokens = self.tokenize(text)
        chunks = []
        start_token = 0
        chunk_id = 0
        
        while start_token < len(tokens):
            end_token = min(start_token + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "start_token": start_token,
                "end_token": end_token
            })
            
            chunk_id += 1
            # Move forward by (chunk_size - overlap) tokens
            start_token += (self.chunk_size - self.overlap)
            
            # If we're at the end, break
            if end_token >= len(tokens):
                break
        
        return chunks
    
    async def build_index(self, source_file: str, output_file: str = "index.json") -> Dict[str, Any]:
        """
        Build index from source file.
        
        Args:
            source_file: Path to source text file (e.g., database.txt)
            output_file: Path to output JSON file (default: index.json)
            
        Returns:
            Index dictionary
        """
        # Read source file
        with open(source_file, 'r', encoding='utf-8') as f:
            source_text = f.read()
        
        # Chunk the text
        chunks = self.chunk_text(source_text)
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.client.embedding(
            input_texts=chunk_texts,
            model_name="azure.text-embedding-3-large",
            dimensions=1536
        )
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
        
        # Calculate total tokens
        total_tokens = len(self.tokenize(source_text))
        
        # Build index structure
        index = {
            "metadata": {
                "source_file": source_file,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "total_chunks": len(chunks),
                "total_tokens": total_tokens
            },
            "chunks": chunks
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        return index
    
    async def add_chunk(self, text: str, index_file: str = "index.json") -> Dict[str, Any]:
        """
        Add a new chunk to the existing index.
        
        Args:
            text: Text to add as a new chunk
            index_file: Path to index JSON file
            
        Returns:
            Updated index dictionary
        """
        # Load existing index
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Index file {index_file} not found")
        
        # Tokenize the new text
        tokens = self.tokenize(text)
        token_count = len(tokens)
        
        # Generate embedding
        embeddings = await self.client.embedding(
            input_texts=[text],
            model_name="azure.text-embedding-3-large",
            dimensions=1536
        )
        
        # Get next chunk_id
        existing_chunk_ids = [chunk["chunk_id"] for chunk in index["chunks"]]
        next_chunk_id = max(existing_chunk_ids) + 1 if existing_chunk_ids else 0
        
        # Create new chunk
        new_chunk = {
            "chunk_id": next_chunk_id,
            "text": text,
            "token_count": token_count,
            "start_token": 0,  # Not applicable for user-added facts
            "end_token": token_count,
            "embedding": embeddings[0]
        }
        
        # Add to index
        index["chunks"].append(new_chunk)
        index["metadata"]["total_chunks"] = len(index["chunks"])
        index["metadata"]["total_tokens"] += token_count
        
        # Save updated index
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        return index


async def main():
    """
    Main function to build index from database.txt
    """
    import asyncio
    from .litellm_client import AsyncLiteLLMClient
    
    client = AsyncLiteLLMClient()
    indexer = Indexer(client)
    
    print("Building index from database.txt...")
    index = await indexer.build_index("database.txt", "index.json")
    print(f"Index built successfully: {index['metadata']['total_chunks']} chunks, {index['metadata']['total_tokens']} tokens")


if __name__ == "__main__":
    asyncio.run(main())

