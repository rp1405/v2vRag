import os
from services.llms.OpenAiChat import OpenAiChat
from services.llms.LlamaChat import LlamaChat
from langchain.chat_models.base import BaseChatModel


class LLMFactory:
    def __init__(self, model_type: str):
        self.model_type = model_type.lower()

    def get_llm(self) -> BaseChatModel:
        if self.model_type == "openai":
            return OpenAiChat()
        elif self.model_type == "llama":
            return LlamaChat()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
