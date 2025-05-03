from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage
from typing import AsyncGenerator, Optional, List
import os
import logging

logger = logging.getLogger(__name__)


class OpenAiChat:
    def __init__(
        self,
        temperature: float = 0.5,
        model_name: str = "gpt-3.5-turbo",
        streaming: bool = True,
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI Chat model.

        Args:
            temperature: Controls randomness in the output (0.0 to 1.0)
            model_name: Name of the OpenAI model to use
            streaming: Whether to stream the response
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.llm = ChatOpenAI(
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            model_name=model_name,
            streaming=streaming,
        )

        if not self.llm.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide it through api_key parameter or OPENAI_API_KEY environment variable."
            )

    async def stream(self, query: str) -> AsyncGenerator[dict, None]:
        """
        Stream the response from the OpenAI model.

        Args:
            query: The input query to send to the model

        Yields:
            dict: Chunks of the model's response

        Raises:
            ValueError: If the query is empty or invalid
            Exception: For any other errors during streaming
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            async for chunk in self.llm.astream(query):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming from OpenAI: {str(e)}")
            raise Exception(f"Failed to stream response from OpenAI: {str(e)}")

    async def get_complete_response(self, query: str) -> str:
        """
        Get the complete response from the OpenAI model.

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
            response = await self.llm.ainvoke(query)  # Use ainvoke instead of agenerate
            return (
                str(response.content) if hasattr(response, "content") else str(response)
            )
        except Exception as e:
            logger.error(f"Error getting response from OpenAI: {str(e)}")
            raise Exception(f"Failed to get response from OpenAI: {str(e)}")
