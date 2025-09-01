import os
import io
import logging
import asyncio
from typing import Optional, Dict, Any
from enum import Enum
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import pygame
import tempfile

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceProfile(Enum):
    """Available voice profiles for Azure TTS"""
    WARM_FEMALE = "en-US-JennyNeural"  # Warm, friendly female voice
    WARM_MALE = "en-US-GuyNeural"      # Warm, friendly male voice
    EMPATHETIC_FEMALE = "en-US-AriaNeural"  # Empathetic female voice
    GENTLE_FEMALE = "en-US-SaraNeural"  # Gentle, soft female voice
    PROFESSIONAL_MALE = "en-US-DavisNeural"  # Professional male voice


class SpeechStyle(Enum):
    """Speech styles for emotional expression"""
    EMPATHETIC = "empathetic"
    FRIENDLY = "friendly"
    CHEERFUL = "cheerful"
    GENTLE = "gentle"
    HOPEFUL = "hopeful"
    SORRY = "sorry"
    UNFRIENDLY = "unfriendly"
    WHISPERING = "whispering"


    