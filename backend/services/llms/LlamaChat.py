from langchain.llms import HuggingFacePipeline
from typing import AsyncGenerator, Optional
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import asyncio

logger = logging.getLogger(__name__)


class ResponseChunk:
    def __init__(self, content: str):
        self.content = content


class LlamaChat:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LlamaChat, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        temperature: float = 0.9,
        model_name: str = "meta-llama/Llama-3.2-1B",
        streaming: bool = True,
        chunk_size: int = 10,  # Number of characters per chunk
    ):
        """
        Initialize Llama Chat model for MacBook M3 with MPS acceleration.
        """
        # Skip initialization if already done
        if LlamaChat._initialized:
            return

        try:
            # Use MPS if available (Apple Silicon GPU)
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            ).to(self.device)

            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # force CPU/mps, since "device=0" is CUDA-only
                max_new_tokens=256,
                temperature=temperature,
                do_sample=True,
                return_full_text=False,
            )

            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            self.chunk_size = chunk_size
            LlamaChat._initialized = True
            logger.info("Model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Llama model: {str(e)}")
            raise Exception(f"Failed to initialize Llama model: {str(e)}")

    async def stream(self, query: str) -> AsyncGenerator[dict, None]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            text = self.get_complete_response(query)
            chunks = [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

            for chunk in chunks:
                yield ResponseChunk(chunk)
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error streaming from Llama: {str(e)}")
            raise Exception(f"Failed to stream response from Llama: {str(e)}")

    async def get_complete_response(self, query: str) -> str:
        """
        Get the complete response from the Llama model.

        Args:
            query: The input query to send to the model

        Returns:
            str: The complete response from the model

        Raises:
            ValueError: If the query is empty or invalid
            Exception: For any other errors during response generation
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            response = self.llm.invoke(query)
            return str(response)
        except Exception as e:
            logger.error(f"Error getting response from Llama: {str(e)}")
            raise Exception(f"Failed to get response from Llama: {str(e)}")
