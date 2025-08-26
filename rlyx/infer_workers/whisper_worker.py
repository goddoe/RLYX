import os
import io
import base64

import torch
import whisper
import numpy as np
import soundfile as sf
from ray import serve
from starlette.requests import Request


NUM_WHISPER_WORKERS = os.environ.get("NUM_WHISPER_WORKERS", 8)

@serve.deployment(num_replicas=NUM_WHISPER_WORKERS,
                  ray_actor_options={"num_gpus": 1})
class WhisperWorker:
    """
    Whisper speech recognition worker for audio transcription.
    Supports multiple audio formats and Whisper model sizes.
    """
    
    def __init__(self, model_size="large"):
        """
        Initialize Whisper model.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        self.model = whisper.load_model(model_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.worker_id = os.getpid()
        print(f"WhisperWorker initialized with model: {model_size} on {self.device}")
    
    def transcribe(self, audio_data, language=None, task="transcribe", **kwargs):
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data as numpy array or file path
            language: Language code (e.g., 'en', 'ko', 'ja'). None for auto-detection
            task: 'transcribe' or 'translate' (translate to English)
            **kwargs: Additional arguments for whisper.transcribe()
        
        Returns:
            Dict with transcription results
        """
        try:
            # Transcribe audio
            result = self.model.transcribe(
                audio_data,
                language=language,
                task=task,
                **kwargs
            )
            
            return {
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": result.get("language", ""),
                "worker_id": self.worker_id
            }
        except Exception as e:
            return {
                "error": str(e),
                "worker_id": self.worker_id
            }
    
    def transcribe_from_base64(self, audio_base64, format="wav", **kwargs):
        """
        Transcribe audio from base64 encoded string.
        
        Args:
            audio_base64: Base64 encoded audio data
            format: Audio format (e.g., 'wav', 'mp3')
            **kwargs: Additional arguments for transcribe()
        
        Returns:
            Dict with transcription results
        """
        try:
            # Decode base64 to bytes
            audio_bytes = base64.b64decode(audio_base64)
            
            # Create a file-like object
            audio_buffer = io.BytesIO(audio_bytes)
            # audio_file.name = f"audio.{format}"

            audio_np, sr = sf.read(audio_buffer, dtype="float32")
            
            # Transcribe
            return self.transcribe(audio_np, **kwargs)
        except Exception as e:
            return {
                "error": f"Failed to decode base64 audio: {str(e)}",
                "worker_id": self.worker_id
            }
    
    def transcribe_batch(self, audio_list, **kwargs):
        """
        Transcribe multiple audio files in batch.
        
        Args:
            audio_list: List of audio data (numpy arrays or file paths)
            **kwargs: Additional arguments for transcribe()
        
        Returns:
            List of transcription results
        """
        results = []
        for audio_data in audio_list:
            result = self.transcribe(audio_data, **kwargs)
            results.append(result)
        return results
    
    async def __call__(self, http_request: Request):
        """
        Handle HTTP requests for transcription.
        
        Expected JSON format:
        {
            "audio": "base64_encoded_audio_string",
            "format": "wav",  # optional, default: "wav"
            "language": "en",  # optional, default: auto-detect
            "task": "transcribe",  # optional, default: "transcribe"
        }
        """
        data = await http_request.json()
        
        audio_base64 = data.get("audio")
        if not audio_base64:
            return {"error": "No audio data provided"}
        
        format = data.get("format", "wav")
        language = data.get("language", None)
        task = data.get("task", "transcribe")
        
        result = self.transcribe_from_base64(
            audio_base64,
            format=format,
            language=language,
            task=task
        )
        
        return result

# Ray Serve Apps
whisper_app = WhisperWorker.bind()
