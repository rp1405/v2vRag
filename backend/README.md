# RAG System Backend

This is a backend service that implements a Retrieval-Augmented Generation (RAG) system. It allows users to upload documents and query them using natural language.

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=your_api_key_here
```

## Running the Server

Start the server with:

```bash
python main.py
```

The server will run on `http://localhost:8000`

## API Endpoints

### Upload Document

- **Endpoint**: `/upload`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Body**: file (the document to process)

### Query Document

- **Endpoint**: `/query`
- **Method**: POST
- **Content-Type**: application/json
- **Body**:

```json
{
  "query": "Your question here"
}
```

## Features

- Document processing and chunking
- Vector storage using Chroma
- OpenAI embeddings and LLM integration
- RESTful API endpoints
- CORS support for frontend integration
