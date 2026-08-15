"""Application configuration.

All secrets are loaded from the environment via a ``.env`` file located in the
project root (see ``.env.example`` for the required variables).

No secret values should ever be hard-coded in this module or anywhere else in
the codebase.
"""

import os

import pyaudio
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralised, immutable-by-convention application settings."""

    # ------------------------------------------------------------------
    # External API credentials (supplied through .env)
    # ------------------------------------------------------------------
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # ------------------------------------------------------------------
    # AI model selections
    # ------------------------------------------------------------------
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    TRANSCRIPTION_MODEL: str = os.getenv("TRANSCRIPTION_MODEL", "whisper-large-v3")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")
    AI_TEMPERATURE: float = float(os.getenv("AI_TEMPERATURE", "0.7"))

    # ------------------------------------------------------------------
    # Audio capture settings
    # ------------------------------------------------------------------
    AUDIO_FORMAT = pyaudio.paInt16
    AUDIO_CHANNELS: int = 1
    AUDIO_RATE: int = 44100
    AUDIO_CHUNK: int = 1024
    PATIENT_RECORD_SECONDS: int = 6
    CHAT_RECORD_SECONDS: int = 5
    PATIENT_AUDIO_FILE: str = "audio.wav"
    CHAT_AUDIO_FILE: str = "chat_audio.wav"

    # ------------------------------------------------------------------
    # Data storage
    # ------------------------------------------------------------------
    CSV_FILE: str = "patient_data.csv"

    # ------------------------------------------------------------------
    # Server behaviour
    # ------------------------------------------------------------------
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    HOST: str = "0.0.0.0"
    PORT: int = 5000