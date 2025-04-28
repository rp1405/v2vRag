from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from prompts.rag_system_prompt import rag_system_prompt
from PyPDF2 import PdfReader
import io
import chardet
import os
from typing import Generator, AsyncGenerator
from services.elevenLabsService import ElevenLabsService
class DocumentContext:
    def __init__(self, role: str, content: str, metadata: dict):
        self.role = role
        self.content = content
        self.metadata = metadata

class RAGService:
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self.context: list[DocumentContext] = []
        self.system_prompt = rag_system_prompt
        self.user_prompt = None
        self.eleven_labs= ElevenLabsService()

    def extract_text_from_pdf(self, file_content):
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    def process_file(self, file_content, filename, user_prompt):
        if filename.lower().endswith('.pdf'):
            content = self.extract_text_from_pdf(file_content)
        else:
            encoding = chardet.detect(file_content)['encoding']
            if not encoding:
                encoding = 'utf-8'
            try:
                content = file_content.decode(encoding)
            except UnicodeDecodeError:
                content = file_content.decode('latin-1')

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(content)

        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )
        
        self.vector_store = Chroma.from_texts(
            chunks, 
            embeddings,
            metadatas=[{"source": filename} for _ in chunks]
        )

        # Initialize the LLM (ChatOpenAI) with streaming
        self.llm = ChatOpenAI(
            temperature=0,
            streaming=True,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.user_prompt = user_prompt

    def generate_query(self, query: str) -> str:
        # Get relevant documents
        docs = self.vector_store.similarity_search(query, k=4)
        
        # Build context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"System: {self.system_prompt}\n\n"
        prompt += f"User: {self.user_prompt}\n\n"
        prompt += f"Context: {context}\n\n"
        
        for document in self.context:
            prompt += f"{document.role.capitalize()}: {document.content}\n\n"

        prompt += f"User: {query}"
        return prompt

    async def query_document_stream_async(self, query: str) -> AsyncGenerator[tuple[str, bytes], None]:
        """
        Stream the response from the LLM asynchronously.
        Returns a tuple of (text_chunk, audio_chunk)
        """
        if not self.llm:
            raise ValueError("Please upload a document first")
        
        self.context.append(DocumentContext(role="user", content=query, metadata={}))
        full_prompt = self.generate_query(query)
        
        response_chunks = []
        current_buffer = ""
        current_len = 0
        chunk_size = 50
        
        async for chunk in self.llm.astream(full_prompt):
            if hasattr(chunk, "content") and chunk.content:
                current_buffer += chunk.content
                current_len += 1
                
                if current_len >= chunk_size:
                    # Get audio for the current buffer
                    audio_chunks = []
                    for audio_chunk in self.eleven_labs.stream_text_to_speech(current_buffer):
                        audio_chunks.append(audio_chunk)
                    
                    # Yield both text and audio
                    yield current_buffer, b''.join(audio_chunks)
                    
                    response_chunks.append(current_buffer)
                    current_buffer = ""
                    current_len = 0
        
        # Handle any remaining text
        if current_buffer:
            audio_chunks = []
            for audio_chunk in self.eleven_labs.stream_text_to_speech(current_buffer):
                audio_chunks.append(audio_chunk)
            
            yield current_buffer, b''.join(audio_chunks)
            response_chunks.append(current_buffer)
        
        complete_response = "".join(response_chunks)
        self.context.append(DocumentContext(role="assistant", content=complete_response, metadata={}))

    def query_document(self, query: str) -> str:
        """
        Get the complete response for a query (non-streaming version).
        """
        if not self.llm:
            raise ValueError("Please upload a document first")
        
        self.context.append(DocumentContext(role="user", content=query, metadata={}))
        full_prompt = self.generate_query(query)
        
        response = self.llm.invoke(full_prompt).content
        
        self.context.append(DocumentContext(role="assistant", content=response, metadata={}))
        
        return response
