from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from services.ragService import RAGService
from services.elevenLabsService import ElevenLabsService
from fastapi import WebSocket
import json
import base64
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = RAGService()
eleven_labs_service = ElevenLabsService()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    audio_data = await file.read()
    text = eleven_labs_service.stream_speech_to_text(audio_data)
    return {"text": text}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_prompt: str = ""):
    try:
        file_content = await file.read()
        rag_service.process_file(file_content, file.filename, user_prompt)
        return {"message": "File processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class QueryRequest(BaseModel):
    query: str
@app.post("/query")
async def query_document(request: QueryRequest):
    try:
        response = rag_service.query_document(request.query)
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
            # Receive query from client
            query = await websocket.receive_text()
            
            # Stream response back to client
            async for text_chunk, audio_chunk in rag_service.query_document_stream_async(query):
                # Send text chunk
                await websocket.send_text(text_chunk)
                
                # Send audio chunk
                await websocket.send_bytes(audio_chunk)
                
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

@app.websocket("/ws/audio-query")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive query from client
            query = await websocket.receive_text()
            audio_data = base64.b64decode(query)
            text_query=eleven_labs_service.stream_speech_to_text(audio_data)
            # Stream response back to client
            async for text_chunk, audio_chunk in rag_service.query_document_stream_async(text_query):
                # Send text chunk
                await websocket.send_text(text_chunk)
                
                # Send audio chunk
                await websocket.send_bytes(audio_chunk)
                
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
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
        reload_delay=1
    ) 