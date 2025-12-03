"""
Async LiteLLM Client
Provides async methods for text completion and embeddings using LiteLLM.
"""

import os
from typing import List, Dict, Any, Optional
from litellm import atext_completion, aembedding


class AsyncLiteLLMClient:
    """
    Async client for LiteLLM API calls.
    
    Automatically configures environment variables for OpenAI API endpoint.
    """
    
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the async LiteLLM client.
        
        Args:
            api_base: The API base URL (default: from OPENAI_API_BASE env var)
            api_key: The API key (default: from OPENAI_API_KEY env var)
        """
        # Read from environment variables if not provided
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_base:
            raise ValueError("OPENAI_API_BASE environment variable must be set")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        os.environ["OPENAI_API_BASE"] = self.api_base
        os.environ["OPENAI_API_KEY"] = self.api_key
    
    async def text_completion(
        self,
        prompt: str,
        model_name: str = "azure.gpt-5",
        max_tokens: int = 15000,
        temperature: float = 0.5,
        **kwargs
    ) -> str:
        """
        Generate text completion asynchronously.
        
        Args:
            prompt: The input prompt text
            model_name: The model to use (default: azure.gpt-5)
            max_tokens: Maximum tokens to generate (default: 15000)
            temperature: Sampling temperature (default: 0.5)
            **kwargs: Additional model kwargs
        
        Returns:
            The generated text response
        """
        model_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        response = await atext_completion(
            prompt=prompt,
            model=f"openai/{model_name}",
            **model_kwargs
        )
        
        # Handle both message.content and text attributes
        choice = response.choices[0]
        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
            return choice.message.content
        elif hasattr(choice, 'text'):
            return choice.text
        else:
            raise ValueError("Unexpected response format from LiteLLM")
    
    async def embedding(
        self,
        input_texts: List[str],
        model_name: str = "azure.text-embedding-3-large",
        dimensions: int = 1536,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings asynchronously.
        
        Args:
            input_texts: List of input texts to embed
            model_name: The embedding model to use (default: azure.text-embedding-3-large)
            dimensions: Embedding dimensions (default: 1536)
            **kwargs: Additional model kwargs
        
        Returns:
            List of embedding vectors (one per input text)
        """
        model_kwargs = {
            "dimensions": dimensions,
            **kwargs
        }
        
        response = await aembedding(
            input=input_texts,
            model=f"openai/{model_name}",
            **model_kwargs
        )
        
        return [item['embedding'] for item in response.data]

