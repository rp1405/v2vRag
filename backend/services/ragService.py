from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from prompts.rag_system_prompt import rag_system_prompt
from PyPDF2 import PdfReader
import io
import chardet
import os
from typing import Generator, AsyncGenerator, Optional, List, Tuple, Union, BinaryIO
from services.elevenLabsService import ElevenLabsService
from services.embeddingsFactory import EmbeddingsFactory
from services.llmFactory import LLMFactory
from contextlib import contextmanager
from prompts.rag_system_prompt import SystemPrompt
from loguru import logger
import time
from services.namedEntityService import NamedEntityService


class DocumentContext:
    def __init__(self, role: str, content: str, metadata: dict):
        self.role = role
        self.content = content
        self.metadata = metadata


class RAGService:
    def __init__(self, embeddings: str, llm: str, chunk_size: int = 50):
        self.vector_store: Optional[Chroma] = None
        self.embeddings = EmbeddingsFactory(embeddings).get_embeddings()
        self.llm = LLMFactory(llm).get_llm()
        self.context: List[DocumentContext] = []
        self.system_prompt = SystemPrompt(llm)
        self.user_prompt: Optional[str] = None
        self.eleven_labs = ElevenLabsService()
        self.chunk_size = chunk_size
        self.model_name = llm
        self.ner_service = NamedEntityService()
        self.chunks = []

    def _fix_leading_punctuation(self, chunks):
        fixed_chunks = []
        for chunk in chunks:
            if fixed_chunks and chunk and chunk[0] in ".!?":
                # Move punctuation to end of previous chunk
                fixed_chunks[-1] += chunk[0]
                chunk = chunk[1:]
            chunk = chunk.strip()
            fixed_chunks.append(chunk)
        return fixed_chunks

    @contextmanager
    def _safe_file_handle(self, file_content: bytes) -> Generator[BinaryIO, None, None]:
        """Safely handle file operations with proper cleanup."""
        file_handle = io.BytesIO(file_content)
        try:
            yield file_handle
        finally:
            file_handle.close()

    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF content with proper error handling."""
        try:
            with self._safe_file_handle(file_content) as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    def process_file(
        self, file_content: bytes, filename: str, user_prompt: str
    ) -> None:
        """Process a file and create vector store with proper error handling."""
        try:
            # Clean up any existing vector store
            if self.vector_store:
                self.vector_store = None

            if filename.lower().endswith(".pdf"):
                content = self.extract_text_from_pdf(file_content)
            else:
                encoding = chardet.detect(file_content)["encoding"]
                if not encoding:
                    encoding = "utf-8"
                try:
                    content = file_content.decode(encoding)
                except UnicodeDecodeError:
                    content = file_content.decode("latin-1")
            self.generate_vector_store(content, filename, user_prompt)
        except Exception as e:
            raise ValueError(f"Failed to process file {filename}: {str(e)}")

    def generate_vector_store(
        self, content: str, filename: str = "default", user_prompt: str = ""
    ) -> None:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ".", "?", "!"],
            )
            chunks = text_splitter.split_text(content)
            chunks = self._fix_leading_punctuation(chunks)
            self.chunks = chunks
            logger.info(f"Using embedding: {self.embeddings}")
            # Create a new vector store with a unique collection name
            collection_name = f"doc_{filename}_{int(time.time())}"
            self.vector_store = Chroma.from_texts(
                chunks,
                self.embeddings,
                metadatas=[{"source": filename} for _ in chunks],
                collection_name=collection_name,
            )

            self.user_prompt = user_prompt
        except Exception as e:
            raise ValueError(f"Failed to generate vector store: {str(e)}")

    def generate_query(self, query: str) -> str:
        """Generate a query with proper error handling."""
        if not self.vector_store:
            raise ValueError(
                "Vector store not initialized. Please process a file first."
            )

        try:
            # Get relevant documents
            similarity_docs = self.vector_store.similarity_search(query, k=4)
            context = "\n\n".join([doc.page_content for doc in similarity_docs])

            keyword_docs, keyword_scores = self.ner_service.search_chunks_for_entities(
                query, self.chunks, k=4
            )
            context = context + "\n\n" + "\n\n".join(keyword_docs)

            prev_context_text = ""
            for document in self.context:
                prev_context_text += (
                    f"{document.role.capitalize()}: {document.content}\n\n"
                )

            prompt = self.system_prompt.get_prompt(
                query, prev_context_text, context, self.user_prompt
            )
            logger.info(f"Prompt: {prompt}")
            return prompt
        except Exception as e:
            raise ValueError(f"Failed to generate query: {str(e)}")

    async def query_document_stream_async(
        self, query: str
    ) -> AsyncGenerator[Tuple[str, bytes], None]:
        """
        Stream the response from the LLM asynchronously with proper error handling.
        Returns a tuple of (text_chunk, audio_chunk)
        """
        if not self.llm:
            raise ValueError("LLM not initialized. Please check your configuration.")

        try:
            full_prompt = self.generate_query(query)
            self.context.append(
                DocumentContext(role="user", content=query, metadata={})
            )

            response_chunks = []
            current_buffer = ""
            current_len = 0

            async for chunk in self.llm.stream(full_prompt):
                if not hasattr(chunk, "content"):
                    continue

                current_buffer += chunk.content
                current_len += 1

                if current_len >= self.chunk_size:
                    try:
                        # Get audio for the current buffer
                        audio_chunks = []
                        for audio_chunk in self.eleven_labs.stream_text_to_speech(
                            current_buffer
                        ):
                            audio_chunks.append(audio_chunk)

                        # Yield both text and audio
                        yield current_buffer, b"".join(audio_chunks)

                        response_chunks.append(current_buffer)
                        current_buffer = ""
                        current_len = 0
                    except Exception as e:
                        raise ValueError(f"Failed to generate audio: {str(e)}")

            # Handle any remaining text
            if current_buffer:
                try:
                    audio_chunks = []
                    for audio_chunk in self.eleven_labs.stream_text_to_speech(
                        current_buffer
                    ):
                        audio_chunks.append(audio_chunk)

                    yield current_buffer, b"".join(audio_chunks)
                    response_chunks.append(current_buffer)
                except Exception as e:
                    raise ValueError(
                        f"Failed to generate audio for remaining text: {str(e)}"
                    )

            complete_response = "".join(response_chunks)
            self.context.append(
                DocumentContext(
                    role="assistant", content=complete_response, metadata={}
                )
            )
        except Exception as e:
            raise ValueError(f"Failed to process query: {str(e)}")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.vector_store:
            self.vector_store = None
        self.context.clear()
        self.user_prompt = None

    async def query_document(self, query: str) -> str:
        """
        Get the complete response for a query (non-streaming version).
        """
        if not self.llm:
            raise ValueError("Please upload a document first")

        full_prompt = self.generate_query(query)
        self.context.append(DocumentContext(role="user", content=query, metadata={}))

        response = await self.llm.get_complete_response(full_prompt)

        self.context.append(
            DocumentContext(role="assistant", content=response, metadata={})
        )

        return response
