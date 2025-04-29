import os
import json
import asyncio
import websockets
import requests
from typing import Generator, Optional, AsyncGenerator
from dotenv import load_dotenv
import urllib3
import certifi

load_dotenv()

class ElevenLabsService:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        self.base_url = "https://api.elevenlabs.io/v1"
        self.output_format = "mp3_44100"
        self.stability = 0.5
        self.similarity_boost = 0.5
        self.model_id = "eleven_turbo_v2_5"
        
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is not set")
        if not self.voice_id:
            raise ValueError("ELEVENLABS_VOICE_ID environment variable is not set")
            
        # Configure requests session with proper SSL settings
        self.session = requests.Session()
        self.session.verify = certifi.where()
        self.session.headers.update({
            "xi-api-key": self.api_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json"
        })

    async def stream_text_to_speech_websocket(
        self,
        text: str,
        voice_id: str,
        model_id: str = "eleven_monolingual_v1",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text to ElevenLabs API using WebSocket and get back audio stream.
        
        Args:
            text (str): The text to convert to speech
            voice_id (str): The ID of the voice to use
            model_id (str): The model ID to use for generation
            stability (float): Voice stability setting (0-1)
            similarity_boost (float): Voice similarity boost setting (0-1)
            style (float): Voice style setting (0-1)
            use_speaker_boost (bool): Whether to use speaker boost
            
        Yields:
            bytes: Chunks of audio data
        """
        headers = {
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": use_speaker_boost
            }
        }
        
        try:
            async with websockets.connect(
                f"{self.ws_url}/{voice_id}",
                extra_headers=headers
            ) as websocket:
                # Send the text data
                await websocket.send(json.dumps(data))
                
                # Receive audio chunks
                while True:
                    try:
                        message = await websocket.recv()
                        if isinstance(message, bytes):
                            yield message
                    except websockets.exceptions.ConnectionClosed:
                        break
                    
        except Exception as e:
            raise Exception(f"Error streaming from ElevenLabs WebSocket: {str(e)}")

    def stream_text_to_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None
    ) -> Generator[bytes, None, None]:
        """
        Stream text to ElevenLabs API and get back audio stream.
        
        Args:
            text (str): The text to convert to speech
            voice_id (str, optional): The ID of the voice to use
            stability (float, optional): Voice stability setting (0-1)
            similarity_boost (float, optional): Voice similarity boost setting (0-1)
            
        Yields:
            bytes: Chunks of audio data
        """
        url = f"{self.base_url}/text-to-speech/{voice_id or self.voice_id}"
        
        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": stability or self.stability,
                "similarity_boost": similarity_boost or self.similarity_boost
            },
            "output_format": self.output_format
        }
        
        try:
            response = self.session.post(url, json=data, stream=True)
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk
                    
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error streaming from ElevenLabs: {str(e)}")

    def get_available_voices(self) -> list:
        """
        Get list of available voices from ElevenLabs.
        
        Returns:
            list: List of available voices
        """
        url = f"{self.base_url}/voices"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()["voices"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error getting voices from ElevenLabs: {str(e)}")
        
    def stream_speech_to_text(self, audio: bytes) -> str:
        """
        Convert speech to text using ElevenLabs API.
        
        Args:
            audio (bytes): The audio data to convert
            
        Returns:
            str: The transcribed text
        """
        url = f"{self.base_url}/speech-to-text"
        
        try:
            # Create a new session for this request
            session = requests.Session()
            session.verify = certifi.where()
            session.headers.update({
                "xi-api-key": self.api_key
            })
            
            # Prepare the multipart form data
            files = {
                'file': ('audio.wav', audio, 'audio/wav')
            }
            data = {
                'model_id': 'scribe_v1'  # or your preferred model
            }
            
            response = session.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()["text"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error converting speech to text: {str(e)}")
        except KeyError as e:
            raise Exception(f"Unexpected response format from ElevenLabs: {str(e)}")
        except Exception as e:
            raise Exception(f"Unknown error in speech to text conversion: {str(e)}")
