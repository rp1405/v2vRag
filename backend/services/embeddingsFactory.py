import os
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    CohereEmbeddings,
)
import torch


class EmbeddingsFactory:
    def __init__(self, embeddings: str):
        self.embeddings = embeddings

    def get_embeddings(self):
        match self.embeddings:
            case "openai":
                return OpenAIEmbeddings(
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    model="text-embedding-ada-002",
                )
            case "bge":
                return HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5",
                    model_kwargs={
                        "device": "mps" if torch.backends.mps.is_available() else "cpu"
                    },
                    encode_kwargs={"normalize_embeddings": True},
                )
            case "huggingface":
                return HuggingFaceEmbeddings()
            case "cohere":
                return CohereEmbeddings()
            case _:
                return OpenAIEmbeddings(
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    model="text-embedding-ada-002",
                )
