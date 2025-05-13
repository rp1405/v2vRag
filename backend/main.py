from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from services.ragService import RAGService
from services.elevenLabsService import ElevenLabsService
from fastapi import WebSocket
import json
import base64
from services.llms.LlamaChat import LlamaChat
import uuid
from datetime import datetime, timedelta
from loguru import logger
from services.scraper import WebScraper

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to store RAG services for different sessions
rag_services: dict[str, RAGService] = {}
# Dictionary to store session expiration times
session_expirations: dict[str, datetime] = {}
# Session timeout in hours
SESSION_TIMEOUT = 24

eleven_labs_service = ElevenLabsService()


def cleanup_expired_sessions():
    current_time = datetime.now()
    expired_sessions = [
        session_id
        for session_id, expiration in session_expirations.items()
        if current_time > expiration
    ]
    for session_id in expired_sessions:
        del rag_services[session_id]
        del session_expirations[session_id]


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    audio_data = await file.read()
    text = eleven_labs_service.stream_speech_to_text(audio_data)
    return {"text": text}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    model: str = Form("openai"),
    embeddings: str = Form("openai"),
    user_prompt: str = Form(""),
    query_language: str = Form("english"),
):
    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())

        # Create new RAG service for this session
        rag_services[session_id] = RAGService(embeddings=embeddings, llm=model)

        # Set session expiration
        session_expirations[session_id] = datetime.now() + timedelta(
            hours=SESSION_TIMEOUT
        )

        # Clean up any expired sessions
        cleanup_expired_sessions()

        file_content = await file.read()
        rag_services[session_id].process_file(
            file_content, file.filename, user_prompt, query_language
        )
        return {"message": "File processed successfully", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ScrapeRequest(BaseModel):
    url: str
    embeddings: str
    model: str
    user_prompt: str = ""


@app.post("/scrape")
async def scrape_url(body: ScrapeRequest):
    try:
        scraper = WebScraper(body.url)
        scraper.scrape()
        final_text = scraper.get_final_text()

        # Generate a unique session ID
        session_id = str(uuid.uuid4())

        # Create new RAG service for this session
        rag_services[session_id] = RAGService(
            embeddings=body.embeddings, llm=body.model
        )

        # Set session expiration
        session_expirations[session_id] = datetime.now() + timedelta(
            hours=SESSION_TIMEOUT
        )

        rag_services[session_id].generate_vector_store(final_text)

        return {
            "message": "Scraped successfully",
            "generated_prompt": final_text,
            "session_id": session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class QueryRequest(BaseModel):
    query: str
    session_id: str


@app.post("/query")
async def query_document(request: QueryRequest):
    try:
        if request.session_id not in rag_services:
            raise HTTPException(
                status_code=400, detail="Invalid session ID or session expired"
            )

        # Update session expiration
        session_expirations[request.session_id] = datetime.now() + timedelta(
            hours=SESSION_TIMEOUT
        )

        response = await rag_services[request.session_id].query_document(request.query)
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/query")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive query and session_id from client
            data = await websocket.receive_json()
            query = data.get("query")
            session_id = data.get("session_id")

            if session_id not in rag_services:
                await websocket.send_text("Invalid session ID or session expired")
                break

            # Update session expiration
            session_expirations[session_id] = datetime.now() + timedelta(
                hours=SESSION_TIMEOUT
            )

            # Stream response back to client
            async for (
                text_chunk,
                audio_chunk,
            ) in rag_services[
                session_id
            ].query_document_stream_async(query):
                # Send text chunk
                await websocket.send_text(text_chunk)

                # Send audio chunk
                await websocket.send_bytes(audio_chunk)

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()


@app.websocket("/ws/audio-query")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive query and session_id from client
            data = await websocket.receive_json()
            audio_data = base64.b64decode(data.get("query"))
            session_id = data.get("session_id")

            if session_id not in rag_services:
                await websocket.send_text("Invalid session ID or session expired")
                break

            # Update session expiration
            session_expirations[session_id] = datetime.now() + timedelta(
                hours=SESSION_TIMEOUT
            )

            text_query = eleven_labs_service.stream_speech_to_text(audio_data)
            # Stream response back to client
            async for (
                text_chunk,
                audio_chunk,
            ) in rag_services[
                session_id
            ].query_document_stream_async(text_query):
                # Send text chunk
                await websocket.send_text(text_chunk)

                # Send audio chunk
                await websocket.send_bytes(audio_chunk)

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."],
        reload_delay=1,
    )
