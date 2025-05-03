import os
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    CohereEmbeddings,
)


class EmbeddingsFactory:
    def __init__(self, embeddings: str):
        self.embeddings = embeddings

    def get_embeddings(self):
        if self.embeddings == "openai":
            return OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-ada-002",
            )
        elif self.embeddings == "huggingface":
            return HuggingFaceEmbeddings()
        elif self.embeddings == "cohere":
            return CohereEmbeddings()
